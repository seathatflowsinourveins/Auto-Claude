"""
Unit Tests for Advanced RAG Patterns: Self-RAG, CRAG, HyDE, and RRF Fusion

Tests cover:
- Self-RAG: Reflection token generation, retrieval decision logic, support grading, iterative refinement
- Corrective RAG (CRAG): Document grading, web fallback triggers, knowledge refinement, ambiguous handling
- HyDE: Hypothetical document generation, embedding averaging, retrieval improvement
- RRF Fusion: K=60 fusion, score normalization

All tests are independent and designed to run fast (<100ms each).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# =============================================================================
# DYNAMIC IMPORT HELPER (bypasses platform module name conflict)
# =============================================================================

def _load_module_from_path(module_name: str, file_path: str):
    """Dynamically load a module from file path to bypass import conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Determine the platform directory path
_tests_dir = Path(__file__).parent
_platform_dir = _tests_dir.parent
_core_rag_dir = _platform_dir / "core" / "rag"

# Pre-load RAG modules to avoid import conflicts with Python's platform module
_self_rag_module = None
_corrective_rag_module = None
_hyde_module = None
_reranker_module = None


def _get_self_rag_module():
    """Lazy load self_rag module."""
    global _self_rag_module
    if _self_rag_module is None:
        _self_rag_module = _load_module_from_path(
            "self_rag_module",
            str(_core_rag_dir / "self_rag.py")
        )
    return _self_rag_module


def _get_corrective_rag_module():
    """Lazy load corrective_rag module."""
    global _corrective_rag_module
    if _corrective_rag_module is None:
        _corrective_rag_module = _load_module_from_path(
            "corrective_rag_module",
            str(_core_rag_dir / "corrective_rag.py")
        )
    return _corrective_rag_module


def _get_hyde_module():
    """Lazy load hyde module."""
    global _hyde_module
    if _hyde_module is None:
        _hyde_module = _load_module_from_path(
            "hyde_module",
            str(_core_rag_dir / "hyde.py")
        )
    return _hyde_module


def _get_reranker_module():
    """Lazy load reranker module."""
    global _reranker_module
    if _reranker_module is None:
        _reranker_module = _load_module_from_path(
            "reranker_module",
            str(_core_rag_dir / "reranker.py")
        )
    return _reranker_module


# =============================================================================
# MOCK PROVIDERS
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider for testing RAG patterns."""

    def __init__(self, responses: Optional[Dict[str, str]] = None, default_response: str = "Mock response"):
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []
        self.default_response = default_response
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.call_count += 1
        self.calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        })

        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response


class MockRetrieverProvider:
    """Mock retriever provider for testing."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self.results = results or []
        self.calls: List[Dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k, **kwargs})

        if self.results:
            return self.results[:top_k]

        return [
            {"content": f"Document about {query}", "metadata": {"id": str(i)}}
            for i in range(min(top_k, 3))
        ]


class MockWebSearchProvider:
    """Mock web search provider for CRAG testing."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self.results = results or []
        self.calls: List[Dict[str, Any]] = []
        self.should_fail = False

    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if self.should_fail:
            raise Exception("Web search failed")

        self.calls.append({"query": query, "max_results": max_results, **kwargs})

        if self.results:
            return self.results[:max_results]

        return [
            {"content": f"Web result for {query}", "url": f"https://example.com/{i}"}
            for i in range(min(max_results, 3))
        ]


class MockEmbeddingProvider:
    """Mock embedding provider for HyDE testing."""

    def __init__(self, embedding_dim: int = 384):
        self._embedding_dim = embedding_dim
        self.calls: List[Any] = []

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        self.calls.append(texts)

        # Return deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            # Create deterministic embedding from text
            seed = hash(text) % 1000
            embedding = [(seed + i) / 1000.0 for i in range(self._embedding_dim)]
            embeddings.append(embedding)

        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class MockVectorStoreProvider:
    """Mock vector store provider for HyDE testing."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self.results = results or []
        self.calls: List[Dict[str, Any]] = []

    async def search(
        self,
        embedding: Any,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        self.calls.append({"embedding_len": len(embedding), "top_k": top_k})

        if self.results:
            return self.results[:top_k]

        return [
            {"content": f"Retrieved document {i}", "metadata": {"score": 0.9 - i * 0.1}}
            for i in range(min(top_k, 3))
        ]


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_retriever():
    """Create a mock retriever provider."""
    return MockRetrieverProvider([
        {"content": "Document 1: Relevant information", "metadata": {"id": "1"}},
        {"content": "Document 2: More context", "metadata": {"id": "2"}},
        {"content": "Document 3: Additional data", "metadata": {"id": "3"}},
    ])


@pytest.fixture
def mock_web_search():
    """Create a mock web search provider."""
    return MockWebSearchProvider([
        {"content": "Web result 1", "url": "https://example.com/1"},
        {"content": "Web result 2", "url": "https://example.com/2"},
    ])


@pytest.fixture
def mock_embedder():
    """Create a mock embedding provider."""
    return MockEmbeddingProvider(embedding_dim=384)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store provider."""
    return MockVectorStoreProvider([
        {"content": "Stored document 1", "metadata": {"id": "vs1"}},
        {"content": "Stored document 2", "metadata": {"id": "vs2"}},
    ])


# =============================================================================
# SELF-RAG TESTS
# =============================================================================

class TestSelfRAG:
    """Tests for Self-RAG implementation."""

    @pytest.mark.asyncio
    async def test_reflection_token_generation(self, mock_llm, mock_retriever):
        """Test that reflection tokens are properly generated."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        ReflectionToken = self_rag_mod.ReflectionToken

        mock_llm.responses["Analyze this query"] = "DECISION: YES\nREASONING: Factual query"
        mock_llm.responses["Evaluate if this document"] = "RELEVANCE: RELEVANT\nSCORE: 0.85\nREASONING: Directly relevant"
        mock_llm.responses["Evaluate if this response"] = "SUPPORT: FULLY_SUPPORTED\nSCORE: 0.9\nUNSUPPORTED_CLAIMS: none"
        mock_llm.responses["Rate how useful"] = "USEFULNESS: 4\nREASONING: Good answer"

        config = SelfRAGConfig(max_iterations=1, enable_iterative_refinement=False)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        result = await self_rag.generate("What is machine learning?")

        assert len(result.reflection_trace) >= 1
        assert any(r.token == ReflectionToken.RETRIEVE for r in result.reflection_trace)

    @pytest.mark.asyncio
    async def test_retrieval_decision_logic_yes(self, mock_llm, mock_retriever):
        """Test retrieval decision when answer is YES."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        RetrievalDecision = self_rag_mod.RetrievalDecision

        mock_llm.responses["Analyze this query"] = "DECISION: YES\nREASONING: Needs external info"

        config = SelfRAGConfig(enable_adaptive_retrieval=True)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        retrieval_decision = await self_rag._decide_retrieval("What are the latest AI trends?")

        assert retrieval_decision.decision == RetrievalDecision.YES.value

    @pytest.mark.asyncio
    async def test_retrieval_decision_logic_no(self, mock_llm, mock_retriever):
        """Test retrieval decision when answer is NO."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        RetrievalDecision = self_rag_mod.RetrievalDecision

        mock_llm.responses["Analyze this query"] = "DECISION: NO\nREASONING: General knowledge"

        config = SelfRAGConfig(enable_adaptive_retrieval=True)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        retrieval_decision = await self_rag._decide_retrieval("What is 2+2?")

        assert retrieval_decision.decision == RetrievalDecision.NO.value

    @pytest.mark.asyncio
    async def test_retrieval_decision_uncertain(self, mock_llm, mock_retriever):
        """Test retrieval decision when answer is UNCERTAIN."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        RetrievalDecision = self_rag_mod.RetrievalDecision

        mock_llm.responses["Analyze this query"] = "DECISION: UNCERTAIN\nREASONING: Could go either way"

        config = SelfRAGConfig(enable_adaptive_retrieval=True)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        retrieval_decision = await self_rag._decide_retrieval("What is the meaning of life?")

        assert retrieval_decision.decision == RetrievalDecision.UNCERTAIN.value

    @pytest.mark.asyncio
    async def test_support_grading_fully_supported(self, mock_llm, mock_retriever):
        """Test support grading when response is fully supported."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        SupportGrade = self_rag_mod.SupportGrade

        mock_llm.responses["Evaluate if this response"] = "SUPPORT: FULLY_SUPPORTED\nSCORE: 0.95\nUNSUPPORTED_CLAIMS: none"

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        support_result = await self_rag._check_support(
            "Machine learning uses data to make predictions.",
            ["Machine learning is a method of data analysis that automates model building."]
        )

        assert support_result.confidence >= 0.9
        assert SupportGrade.FULLY_SUPPORTED.value in support_result.decision

    @pytest.mark.asyncio
    async def test_support_grading_partially_supported(self, mock_llm, mock_retriever):
        """Test support grading when response is partially supported."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        SupportGrade = self_rag_mod.SupportGrade

        mock_llm.responses["Evaluate if this response"] = "SUPPORT: PARTIALLY_SUPPORTED\nSCORE: 0.6\nUNSUPPORTED_CLAIMS: claim about speed"

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        support_result = await self_rag._check_support(
            "ML is fast and accurate.",
            ["Machine learning can be accurate."]
        )

        assert 0.5 <= support_result.confidence <= 0.7
        assert SupportGrade.PARTIALLY_SUPPORTED.value in support_result.decision

    @pytest.mark.asyncio
    async def test_support_grading_not_supported(self, mock_llm, mock_retriever):
        """Test support grading when response is not supported."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        SupportGrade = self_rag_mod.SupportGrade

        mock_llm.responses["Evaluate if this response"] = "SUPPORT: NOT_SUPPORTED\nSCORE: 0.1\nUNSUPPORTED_CLAIMS: all claims"

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        support_result = await self_rag._check_support(
            "Cats are the best programmers.",
            ["Python is a programming language."]
        )

        assert support_result.confidence <= 0.3
        assert SupportGrade.NOT_SUPPORTED.value in support_result.decision

    @pytest.mark.asyncio
    async def test_support_grading_no_context(self, mock_llm, mock_retriever):
        """Test support grading with no context provided."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig
        SupportGrade = self_rag_mod.SupportGrade

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        support_result = await self_rag._check_support("Any response", [])

        assert support_result.confidence == 0.0
        assert SupportGrade.NOT_SUPPORTED.value in support_result.decision

    @pytest.mark.asyncio
    async def test_iterative_refinement_triggers(self, mock_llm, mock_retriever):
        """Test that iterative refinement triggers when support is low."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        # First iteration: low support -> trigger refinement
        # Second iteration: high support -> complete
        call_count = [0]

        async def dynamic_generate(prompt, **kwargs):
            call_count[0] += 1
            if "Evaluate if this response" in prompt:
                if call_count[0] <= 4:  # First evaluation
                    return "SUPPORT: PARTIALLY_SUPPORTED\nSCORE: 0.3\nUNSUPPORTED_CLAIMS: incomplete"
                else:  # After refinement
                    return "SUPPORT: FULLY_SUPPORTED\nSCORE: 0.9\nUNSUPPORTED_CLAIMS: none"
            elif "Rate how useful" in prompt:
                if call_count[0] <= 5:
                    return "USEFULNESS: 2\nREASONING: Needs improvement"
                else:
                    return "USEFULNESS: 5\nREASONING: Excellent"
            elif "Improve this response" in prompt:
                return "Improved response with better support"
            return "Default response"

        mock_llm.generate = dynamic_generate
        mock_llm.responses["Analyze this query"] = "DECISION: NO\nREASONING: Skip retrieval"

        config = SelfRAGConfig(
            max_iterations=3,
            support_threshold=0.6,
            usefulness_threshold=3,
            enable_iterative_refinement=True,
            enable_adaptive_retrieval=False
        )
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        result = await self_rag.generate("Test query")

        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_iterative_refinement_max_iterations(self, mock_llm, mock_retriever):
        """Test that refinement stops at max iterations."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        # Always return low scores to trigger refinement
        mock_llm.responses["Evaluate if this response"] = "SUPPORT: NOT_SUPPORTED\nSCORE: 0.1"
        mock_llm.responses["Rate how useful"] = "USEFULNESS: 1\nREASONING: Poor"
        mock_llm.responses["Analyze this query"] = "DECISION: NO"

        config = SelfRAGConfig(
            max_iterations=2,
            support_threshold=0.9,
            usefulness_threshold=5,
            enable_iterative_refinement=True,
            enable_adaptive_retrieval=False
        )
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        result = await self_rag.generate("Test query")

        assert result.iterations <= 2

    @pytest.mark.asyncio
    async def test_usefulness_scoring(self, mock_llm, mock_retriever):
        """Test usefulness scoring functionality."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        mock_llm.responses["Rate how useful"] = "USEFULNESS: 5\nREASONING: Very comprehensive and helpful"

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        usefulness_result = await self_rag._check_usefulness(
            "What is RAG?",
            "RAG is Retrieval-Augmented Generation, combining retrieval with generation."
        )

        assert usefulness_result.decision == "5"
        assert usefulness_result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_calculation_with_retrieval(self, mock_llm, mock_retriever):
        """Test confidence calculation when retrieval is used."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        confidence = self_rag._calculate_confidence(
            support_score=0.9,
            usefulness_score=5,
            relevant_docs=3,
            retrieval_used=True
        )

        # With retrieval: support*0.5 + usefulness*0.3 + docs*0.2
        # 0.9*0.5 + 1.0*0.3 + 1.0*0.2 = 0.45 + 0.3 + 0.2 = 0.95
        assert 0.9 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_calculation_without_retrieval(self, mock_llm, mock_retriever):
        """Test confidence calculation when retrieval is not used."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        config = SelfRAGConfig()
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        confidence = self_rag._calculate_confidence(
            support_score=0.5,
            usefulness_score=4,
            relevant_docs=0,
            retrieval_used=False
        )

        # Without retrieval: support*0.3 + usefulness*0.7
        # 0.5*0.3 + 0.8*0.7 = 0.15 + 0.56 = 0.71
        assert 0.6 <= confidence <= 0.8

    @pytest.mark.asyncio
    async def test_generate_with_provided_context(self, mock_llm, mock_retriever):
        """Test generation with pre-provided context."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        mock_llm.responses["Answer the query"] = "Response based on provided context"
        mock_llm.responses["Evaluate if this response"] = "SUPPORT: FULLY_SUPPORTED\nSCORE: 0.9"
        mock_llm.responses["Rate how useful"] = "USEFULNESS: 4"

        config = SelfRAGConfig(enable_iterative_refinement=False)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        result = await self_rag.generate(
            "What is X?",
            context=["X is a value", "X is important"]
        )

        assert result.retrieval_used is True
        assert result.relevant_documents == 2
        assert len(mock_retriever.calls) == 0  # Should not call retriever


# =============================================================================
# CORRECTIVE RAG (CRAG) TESTS
# =============================================================================

class TestCorrectiveRAG:
    """Tests for Corrective RAG (CRAG) implementation."""

    @pytest.mark.asyncio
    async def test_document_grading_correct(self, mock_llm, mock_retriever):
        """Test document grading when documents are correct."""
        crag_mod = _get_corrective_rag_module()
        DocumentGrader = crag_mod.DocumentGrader
        GradingDecision = crag_mod.GradingDecision

        mock_llm.responses["Evaluate if this document"] = "GRADE: CORRECT\nSCORE: 0.9\nREASONING: Highly relevant"

        grader = DocumentGrader(llm=mock_llm, correct_threshold=0.7)

        documents = [
            {"content": "Relevant document 1", "metadata": {}},
            {"content": "Relevant document 2", "metadata": {}},
        ]

        result = await grader.grade_documents("What is machine learning?", documents)

        assert result.decision == GradingDecision.CORRECT
        assert result.correct_count >= 1
        assert result.overall_score > 0.7

    @pytest.mark.asyncio
    async def test_document_grading_incorrect(self, mock_llm, mock_retriever):
        """Test document grading when documents are incorrect."""
        crag_mod = _get_corrective_rag_module()
        DocumentGrader = crag_mod.DocumentGrader
        GradingDecision = crag_mod.GradingDecision

        mock_llm.responses["Evaluate if this document"] = "GRADE: INCORRECT\nSCORE: 0.1\nREASONING: Not relevant"

        grader = DocumentGrader(llm=mock_llm, correct_threshold=0.7)

        documents = [
            {"content": "Irrelevant content", "metadata": {}},
        ]

        result = await grader.grade_documents("What is machine learning?", documents)

        assert result.decision == GradingDecision.INCORRECT
        assert result.incorrect_count >= 1
        assert result.overall_score < 0.3

    @pytest.mark.asyncio
    async def test_document_grading_ambiguous(self, mock_llm, mock_retriever):
        """Test document grading when documents are ambiguous."""
        crag_mod = _get_corrective_rag_module()
        DocumentGrader = crag_mod.DocumentGrader
        GradingDecision = crag_mod.GradingDecision

        mock_llm.responses["Evaluate if this document"] = "GRADE: AMBIGUOUS\nSCORE: 0.5\nREASONING: Partially relevant"

        grader = DocumentGrader(llm=mock_llm, correct_threshold=0.7, ambiguous_threshold=0.3)

        documents = [
            {"content": "Somewhat related content", "metadata": {}},
        ]

        result = await grader.grade_documents("What is AI?", documents)

        assert result.decision == GradingDecision.AMBIGUOUS
        assert result.ambiguous_count >= 1

    @pytest.mark.asyncio
    async def test_document_grading_empty_documents(self, mock_llm):
        """Test document grading with empty document list."""
        crag_mod = _get_corrective_rag_module()
        DocumentGrader = crag_mod.DocumentGrader
        GradingDecision = crag_mod.GradingDecision

        grader = DocumentGrader(llm=mock_llm)

        result = await grader.grade_documents("Any query", [])

        assert result.decision == GradingDecision.INCORRECT
        assert result.overall_score == 0.0
        assert len(result.graded_documents) == 0

    @pytest.mark.asyncio
    async def test_web_fallback_trigger_on_incorrect(self, mock_llm, mock_retriever, mock_web_search):
        """Test that web search is triggered when documents are incorrect."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig
        GradingDecision = crag_mod.GradingDecision

        mock_llm.responses["Evaluate if this document"] = "GRADE: INCORRECT\nSCORE: 0.1\nREASONING: Not relevant"
        mock_llm.responses["Rewrite this query"] = "optimized search query"
        mock_llm.responses["Answer the query"] = "Generated response from web"

        config = CRAGConfig(enable_web_fallback=True)
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("Latest AI developments?")

        assert len(mock_web_search.calls) >= 1
        assert result.source_type == "web_search"
        assert result.documents_from_web > 0

    @pytest.mark.asyncio
    async def test_web_fallback_not_triggered_on_correct(self, mock_llm, mock_retriever, mock_web_search):
        """Test that web search is not triggered when documents are correct."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: CORRECT\nSCORE: 0.9\nREASONING: Highly relevant"
        mock_llm.responses["Answer the query"] = "Generated response from retriever"

        config = CRAGConfig(enable_web_fallback=True)
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("What is machine learning?")

        assert len(mock_web_search.calls) == 0
        assert result.source_type == "retriever"
        assert result.documents_from_web == 0

    @pytest.mark.asyncio
    async def test_web_fallback_disabled(self, mock_llm, mock_retriever, mock_web_search):
        """Test that web search is not triggered when disabled."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: INCORRECT\nSCORE: 0.1"
        mock_llm.responses["Answer the query"] = "Response without web"

        config = CRAGConfig(enable_web_fallback=False)
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("Test query")

        assert len(mock_web_search.calls) == 0
        assert result.source_type == "retriever"

    @pytest.mark.asyncio
    async def test_knowledge_refinement_on_ambiguous(self, mock_llm, mock_retriever, mock_web_search):
        """Test knowledge refinement is applied for ambiguous documents."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig
        KnowledgeRefiner = crag_mod.KnowledgeRefiner

        mock_llm.responses["Evaluate if this document"] = "GRADE: AMBIGUOUS\nSCORE: 0.5\nREASONING: Partial"
        mock_llm.responses["Extract and refine"] = "Refined knowledge from partial documents"
        mock_llm.responses["Answer the query"] = "Response using refined knowledge"

        config = CRAGConfig(enable_knowledge_refinement=True, enable_web_fallback=False)
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("Test query with ambiguous docs")

        assert result.refinement_applied is True

    @pytest.mark.asyncio
    async def test_knowledge_refinement_disabled(self, mock_llm, mock_retriever, mock_web_search):
        """Test knowledge refinement when disabled."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: AMBIGUOUS\nSCORE: 0.5"
        mock_llm.responses["Answer the query"] = "Response without refinement"

        config = CRAGConfig(enable_knowledge_refinement=False, enable_web_fallback=False)
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("Test query")

        assert result.refinement_applied is False

    @pytest.mark.asyncio
    async def test_ambiguous_handling_mixed_sources(self, mock_llm, mock_retriever, mock_web_search):
        """Test handling of ambiguous documents with mixed sources."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: AMBIGUOUS\nSCORE: 0.5"
        mock_llm.responses["Extract and refine"] = "Refined knowledge"
        mock_llm.responses["Rewrite this query"] = "better query"
        mock_llm.responses["Answer the query"] = "Mixed source response"

        config = CRAGConfig(
            enable_knowledge_refinement=True,
            enable_web_fallback=True
        )
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("Ambiguous topic query")

        assert result.source_type == "mixed"
        assert result.refinement_applied is True
        assert result.documents_from_web > 0

    @pytest.mark.asyncio
    async def test_query_rewriting_for_web(self, mock_llm, mock_retriever, mock_web_search):
        """Test query rewriting for web search."""
        crag_mod = _get_corrective_rag_module()
        QueryRewriter = crag_mod.QueryRewriter

        mock_llm.responses["Rewrite this query"] = "optimized search query for web"

        rewriter = QueryRewriter(llm=mock_llm)
        rewritten = await rewriter.rewrite_for_web("What is ML?")

        assert "optimized" in rewritten or len(rewritten) > 0

    @pytest.mark.asyncio
    async def test_grading_parallel_execution(self, mock_llm):
        """Test that document grading runs in parallel."""
        crag_mod = _get_corrective_rag_module()
        DocumentGrader = crag_mod.DocumentGrader

        call_times = []

        original_generate = mock_llm.generate

        async def tracking_generate(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)  # Small delay to detect parallelism
            return "GRADE: CORRECT\nSCORE: 0.9\nREASONING: Good"

        mock_llm.generate = tracking_generate

        grader = DocumentGrader(llm=mock_llm)

        documents = [
            {"content": f"Document {i}", "metadata": {}}
            for i in range(3)
        ]

        await grader.grade_documents("Test query", documents)

        # If parallel, calls should be close in time
        if len(call_times) >= 2:
            time_diff = max(call_times) - min(call_times)
            assert time_diff < 0.05  # Should start within 50ms of each other

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, mock_llm, mock_retriever, mock_web_search):
        """Test confidence calculation in CRAG."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: CORRECT\nSCORE: 0.9"
        mock_llm.responses["Answer the query"] = "High quality response"

        config = CRAGConfig()
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            config=config
        )

        result = await crag.generate("Test query")

        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence > 0.5  # Should be reasonably confident with correct docs


# =============================================================================
# HYDE TESTS
# =============================================================================

class TestHyDE:
    """Tests for HyDE (Hypothetical Document Embeddings) implementation."""

    @pytest.mark.asyncio
    async def test_hypothetical_generation(self, mock_llm, mock_embedder, mock_vector_store):
        """Test hypothetical document generation."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.responses["Write a detailed"] = "Hypothetical answer about machine learning techniques"

        config = HyDEConfig(n_hypothetical=3, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("What is machine learning?")

        assert len(result.hypotheticals) >= 1
        assert all(h.content for h in result.hypotheticals)
        assert mock_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_hypothetical_generation_multiple(self, mock_llm, mock_embedder, mock_vector_store):
        """Test generation of multiple hypothetical documents."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Generated hypothetical document"

        config = HyDEConfig(n_hypothetical=5, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("Complex query")

        assert len(result.hypotheticals) == 5

    @pytest.mark.asyncio
    async def test_embedding_averaging(self, mock_llm, mock_embedder, mock_vector_store):
        """Test embedding averaging strategy."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Hypothetical content"

        config = HyDEConfig(
            n_hypothetical=3,
            embedding_strategy="average",
            enable_caching=False
        )
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("Test query")

        assert result.query_embedding is not None
        assert len(result.query_embedding) == mock_embedder.embedding_dim
        assert len(mock_vector_store.calls) >= 1

    @pytest.mark.asyncio
    async def test_embedding_first_strategy(self, mock_llm, mock_embedder, mock_vector_store):
        """Test 'first' embedding strategy."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Hypothetical content"

        config = HyDEConfig(
            n_hypothetical=3,
            embedding_strategy="first",
            enable_caching=False
        )
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("Test query")

        assert result.query_embedding is not None
        assert len(result.query_embedding) == mock_embedder.embedding_dim

    @pytest.mark.asyncio
    async def test_retrieval_improvement(self, mock_llm, mock_embedder, mock_vector_store):
        """Test that HyDE improves retrieval by using hypotheticals."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Detailed hypothetical answer about the topic"

        config = HyDEConfig(n_hypothetical=2, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("What is RAG?", top_k=5)

        assert len(result.documents) > 0
        assert result.retrieval_count > 0
        assert len(mock_vector_store.calls) >= 1
        # Verify embedding was passed to vector store
        assert mock_vector_store.calls[0]["embedding_len"] == mock_embedder.embedding_dim

    @pytest.mark.asyncio
    async def test_hypothetical_with_domain_hint(self, mock_llm, mock_embedder, mock_vector_store):
        """Test hypothetical generation with domain hint."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Domain-specific hypothetical"

        config = HyDEConfig(
            n_hypothetical=1,
            domain_hint="artificial intelligence",
            enable_caching=False
        )
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        await hyde.retrieve("What is neural network?")

        # Check that domain context was included in prompt
        assert any("artificial intelligence" in call["prompt"] for call in mock_llm.calls)

    @pytest.mark.asyncio
    async def test_include_original_query(self, mock_llm, mock_embedder, mock_vector_store):
        """Test including original query in search."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Hypothetical content"
        mock_vector_store.results = [
            {"content": f"Result {i}", "metadata": {"id": str(i)}}
            for i in range(5)
        ]

        config = HyDEConfig(
            n_hypothetical=2,
            include_original_query=True,
            enable_caching=False
        )
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("Test query")

        # Should have called vector store at least twice (hypothetical + original)
        assert len(mock_vector_store.calls) >= 2

    @pytest.mark.asyncio
    async def test_caching_enabled(self, mock_llm, mock_embedder, mock_vector_store):
        """Test that caching works when enabled."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Cached hypothetical"

        config = HyDEConfig(n_hypothetical=2, enable_caching=True)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        # First call
        result1 = await hyde.retrieve("Cached query")
        llm_calls_first = mock_llm.call_count

        # Second call (should use cache)
        result2 = await hyde.retrieve("Cached query")
        llm_calls_second = mock_llm.call_count

        # LLM should not be called again for same query
        assert llm_calls_second == llm_calls_first

    @pytest.mark.asyncio
    async def test_fallback_on_no_hypotheticals(self, mock_llm, mock_embedder, mock_vector_store):
        """Test fallback when no hypotheticals are generated."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        # Make LLM fail
        async def failing_generate(*args, **kwargs):
            raise Exception("LLM error")

        mock_llm.generate = failing_generate

        config = HyDEConfig(n_hypothetical=2, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("Query with fallback")

        # Should still return results using direct query embedding
        assert result.documents is not None
        assert len(result.hypotheticals) == 0

    @pytest.mark.asyncio
    async def test_hypothesis_type_document(self, mock_llm, mock_embedder, mock_vector_store):
        """Test DOCUMENT hypothesis type."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig
        HypotheticalDocumentType = hyde_mod.HypotheticalDocumentType

        mock_llm.default_response = "Document-style hypothetical"

        config = HyDEConfig(
            n_hypothetical=1,
            hypothesis_type=HypotheticalDocumentType.DOCUMENT,
            enable_caching=False
        )
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        await hyde.retrieve("Test query")

        # Check document-style prompt was used
        assert any("passage from a document" in call["prompt"].lower() for call in mock_llm.calls)

    @pytest.mark.asyncio
    async def test_hypothesis_type_explanation(self, mock_llm, mock_embedder, mock_vector_store):
        """Test EXPLANATION hypothesis type."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig
        HypotheticalDocumentType = hyde_mod.HypotheticalDocumentType

        mock_llm.default_response = "Explanation-style hypothetical"

        config = HyDEConfig(
            n_hypothetical=1,
            hypothesis_type=HypotheticalDocumentType.EXPLANATION,
            enable_caching=False
        )
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        await hyde.retrieve("Test query")

        # Check explanation-style prompt was used
        assert any("detailed explanation" in call["prompt"].lower() for call in mock_llm.calls)


# =============================================================================
# RRF FUSION TESTS
# =============================================================================

class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion (RRF) implementation."""

    @pytest.mark.asyncio
    async def test_k60_fusion_single_list(self):
        """Test RRF fusion with k=60 for a single list."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        docs = [
            Document(id="1", content="First document"),
            Document(id="2", content="Second document"),
            Document(id="3", content="Third document"),
        ]

        result = await reranker.rrf_fusion([docs], k=60)

        assert len(result) == 3
        # First document should have highest RRF score: 1/(60+1) = 0.01639
        assert result[0].document.id == "1"
        assert result[0].score > result[1].score > result[2].score

    @pytest.mark.asyncio
    async def test_k60_fusion_multiple_lists(self):
        """Test RRF fusion with k=60 for multiple lists."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        list1 = [
            Document(id="1", content="Doc 1"),
            Document(id="2", content="Doc 2"),
            Document(id="3", content="Doc 3"),
        ]
        list2 = [
            Document(id="2", content="Doc 2"),  # Appears in both lists
            Document(id="4", content="Doc 4"),
            Document(id="1", content="Doc 1"),  # Appears in both lists
        ]

        result = await reranker.rrf_fusion([list1, list2], k=60)

        # Doc 2 should rank high (rank 1 in list2, rank 2 in list1)
        # Doc 1 should also rank high (rank 1 in list1, rank 3 in list2)
        doc_ids = [r.document.id for r in result]
        assert "1" in doc_ids[:2] or "2" in doc_ids[:2]

    @pytest.mark.asyncio
    async def test_k60_fusion_deduplication(self):
        """Test that RRF fusion properly deduplicates documents."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        list1 = [Document(id="1", content="Content 1")]
        list2 = [Document(id="1", content="Content 1")]  # Same doc
        list3 = [Document(id="1", content="Content 1")]  # Same doc again

        result = await reranker.rrf_fusion([list1, list2, list3], k=60)

        assert len(result) == 1
        # Score should combine from all three lists
        # 3 * (1/(60+1)) = 3 * 0.01639 = 0.04918
        assert abs(result[0].score - 3 / 61) < 0.001

    @pytest.mark.asyncio
    async def test_k60_fusion_empty_lists(self):
        """Test RRF fusion with empty input."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker

        reranker = SemanticReranker(enable_cache=False)

        result = await reranker.rrf_fusion([], k=60)

        assert result == []

    @pytest.mark.asyncio
    async def test_k60_fusion_formula(self):
        """Test that RRF formula is correctly implemented."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)
        k = 60

        list1 = [
            Document(id="A", content="Doc A"),  # rank 1
            Document(id="B", content="Doc B"),  # rank 2
        ]

        result = await reranker.rrf_fusion([list1], k=k)

        # RRF score = 1/(k + rank)
        expected_score_A = 1 / (k + 1)  # 1/61
        expected_score_B = 1 / (k + 2)  # 1/62

        assert abs(result[0].score - expected_score_A) < 0.0001
        assert abs(result[1].score - expected_score_B) < 0.0001

    @pytest.mark.asyncio
    async def test_score_normalization_in_reranking(self):
        """Test score normalization in cross-encoder reranking."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        documents = [
            Document(id="1", content="Highly relevant content about machine learning"),
            Document(id="2", content="Somewhat related content"),
            Document(id="3", content="Completely unrelated topic"),
        ]

        # This will use TF-IDF fallback since cross-encoder not available in test
        result = await reranker.rerank("machine learning", documents, top_k=3)

        # Scores should be normalized between 0 and 1
        for doc in result:
            assert 0.0 <= doc.score <= 1.0

        # Results should be sorted by score descending
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_weighted_rrf_fusion(self):
        """Test RRF fusion with custom weights."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        list1 = [Document(id="A", content="Doc A")]
        list2 = [Document(id="B", content="Doc B")]

        # Give list2 double weight
        result = await reranker.rrf_fusion([list1, list2], k=60, weights=[1.0, 2.0])

        # Doc B should have higher score due to weight
        doc_a_score = next(r.score for r in result if r.document.id == "A")
        doc_b_score = next(r.score for r in result if r.document.id == "B")

        assert doc_b_score > doc_a_score

    @pytest.mark.asyncio
    async def test_rrf_fusion_different_k_values(self):
        """Test RRF fusion with different k values."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        docs = [
            Document(id="1", content="First"),
            Document(id="2", content="Second"),
        ]

        # Lower k = more emphasis on top ranks
        result_low_k = await reranker.rrf_fusion([docs], k=10)
        # Higher k = more smoothing
        result_high_k = await reranker.rrf_fusion([docs], k=100)

        # With lower k, the gap between scores should be larger
        gap_low_k = result_low_k[0].score - result_low_k[1].score
        gap_high_k = result_high_k[0].score - result_high_k[1].score

        assert gap_low_k > gap_high_k

    @pytest.mark.asyncio
    async def test_rrf_fusion_preserves_document_content(self):
        """Test that RRF fusion preserves document content and metadata."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        original_content = "Original document content"
        original_metadata = {"key": "value", "important": True}

        docs = [Document(id="1", content=original_content, metadata=original_metadata)]

        result = await reranker.rrf_fusion([docs], k=60)

        assert result[0].document.content == original_content
        assert result[0].document.metadata == original_metadata

    @pytest.mark.asyncio
    async def test_rrf_metadata_tracking(self):
        """Test that RRF fusion adds appropriate metadata."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        list1 = [Document(id="1", content="Doc 1")]
        list2 = [Document(id="2", content="Doc 2")]

        result = await reranker.rrf_fusion([list1, list2], k=60)

        # Check metadata contains fusion information
        for doc in result:
            assert "fusion_method" in doc.metadata
            assert doc.metadata["fusion_method"] == "rrf"
            assert "k" in doc.metadata
            assert doc.metadata["k"] == 60


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRAGPatternsIntegration:
    """Integration tests combining multiple RAG patterns."""

    @pytest.mark.asyncio
    async def test_self_rag_end_to_end(self, mock_llm, mock_retriever):
        """Test Self-RAG end-to-end flow."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        mock_llm.responses["Analyze this query"] = "DECISION: YES\nREASONING: Factual"
        mock_llm.responses["Evaluate if this document"] = "RELEVANCE: RELEVANT\nSCORE: 0.9"
        mock_llm.responses["Answer the query"] = "Comprehensive answer about the topic"
        mock_llm.responses["Evaluate if this response"] = "SUPPORT: FULLY_SUPPORTED\nSCORE: 0.9"
        mock_llm.responses["Rate how useful"] = "USEFULNESS: 5\nREASONING: Excellent"

        config = SelfRAGConfig(max_iterations=1, enable_iterative_refinement=False)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        result = await self_rag.generate("What is artificial intelligence?")

        assert result.response
        assert result.confidence > 0
        assert len(result.generation_attempts) >= 1

    @pytest.mark.asyncio
    async def test_crag_end_to_end(self, mock_llm, mock_retriever, mock_web_search):
        """Test CRAG end-to-end flow."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: CORRECT\nSCORE: 0.85"
        mock_llm.responses["Answer the query"] = "Answer based on retrieved documents"

        config = CRAGConfig()
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=mock_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("What is machine learning?")

        assert result.response
        assert result.confidence > 0
        assert result.grading_decision is not None

    @pytest.mark.asyncio
    async def test_hyde_end_to_end(self, mock_llm, mock_embedder, mock_vector_store):
        """Test HyDE end-to-end flow."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Hypothetical document about the query topic"

        config = HyDEConfig(n_hypothetical=2, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("What is deep learning?", top_k=5)

        assert len(result.documents) > 0
        assert len(result.hypotheticals) >= 1
        assert result.query_embedding is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestRAGPatternEdgeCases:
    """Tests for edge cases in RAG patterns."""

    @pytest.mark.asyncio
    async def test_self_rag_empty_query(self, mock_llm, mock_retriever):
        """Test Self-RAG with empty query."""
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        mock_llm.responses["Analyze this query"] = "DECISION: NO"
        mock_llm.default_response = "Response for empty query"

        config = SelfRAGConfig(enable_iterative_refinement=False)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        result = await self_rag.generate("")

        assert result is not None

    @pytest.mark.asyncio
    async def test_crag_retriever_failure(self, mock_llm, mock_web_search):
        """Test CRAG when retriever fails."""
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        failing_retriever = MockRetrieverProvider()

        async def fail(*args, **kwargs):
            raise Exception("Retriever error")

        failing_retriever.retrieve = fail

        mock_llm.responses["Rewrite this query"] = "rewritten query"
        mock_llm.responses["Answer the query"] = "Fallback response"

        config = CRAGConfig(enable_web_fallback=True)
        crag = CorrectiveRAG(
            llm=mock_llm,
            retriever=failing_retriever,
            web_search=mock_web_search,
            config=config
        )

        result = await crag.generate("Test query")

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_hyde_embedder_dimension_mismatch(self, mock_llm, mock_vector_store):
        """Test HyDE handles embedding dimension correctly."""
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        # Create embedder with specific dimension
        embedder = MockEmbeddingProvider(embedding_dim=768)
        mock_llm.default_response = "Hypothetical content"

        config = HyDEConfig(n_hypothetical=1, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=embedder,
            vector_store=mock_vector_store,
            config=config
        )

        result = await hyde.retrieve("Test query")

        assert len(result.query_embedding) == 768

    @pytest.mark.asyncio
    async def test_rrf_fusion_mismatched_weights(self):
        """Test RRF fusion with mismatched weights raises error."""
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        list1 = [Document(id="1", content="Doc 1")]
        list2 = [Document(id="2", content="Doc 2")]

        with pytest.raises(ValueError):
            await reranker.rrf_fusion([list1, list2], weights=[1.0])  # Wrong number of weights


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestRAGPatternPerformance:
    """Performance tests ensuring operations complete quickly."""

    @pytest.mark.asyncio
    async def test_self_rag_fast_execution(self, mock_llm, mock_retriever):
        """Test Self-RAG completes within time limit."""
        import time
        self_rag_mod = _get_self_rag_module()
        SelfRAG = self_rag_mod.SelfRAG
        SelfRAGConfig = self_rag_mod.SelfRAGConfig

        mock_llm.responses["Analyze this query"] = "DECISION: NO"
        mock_llm.default_response = "Quick response"

        config = SelfRAGConfig(max_iterations=1, enable_iterative_refinement=False)
        self_rag = SelfRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        start = time.time()
        await self_rag.generate("Quick test")
        duration = time.time() - start

        assert duration < 0.1  # Should complete in under 100ms

    @pytest.mark.asyncio
    async def test_crag_fast_execution(self, mock_llm, mock_retriever):
        """Test CRAG completes within time limit."""
        import time
        crag_mod = _get_corrective_rag_module()
        CorrectiveRAG = crag_mod.CorrectiveRAG
        CRAGConfig = crag_mod.CRAGConfig

        mock_llm.responses["Evaluate if this document"] = "GRADE: CORRECT\nSCORE: 0.9"
        mock_llm.default_response = "Quick response"

        config = CRAGConfig(enable_web_fallback=False)
        crag = CorrectiveRAG(llm=mock_llm, retriever=mock_retriever, config=config)

        start = time.time()
        await crag.generate("Quick test")
        duration = time.time() - start

        assert duration < 0.1

    @pytest.mark.asyncio
    async def test_hyde_fast_execution(self, mock_llm, mock_embedder, mock_vector_store):
        """Test HyDE completes within time limit."""
        import time
        hyde_mod = _get_hyde_module()
        HyDERetriever = hyde_mod.HyDERetriever
        HyDEConfig = hyde_mod.HyDEConfig

        mock_llm.default_response = "Quick hypothetical"

        config = HyDEConfig(n_hypothetical=1, enable_caching=False)
        hyde = HyDERetriever(
            llm=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config
        )

        start = time.time()
        await hyde.retrieve("Quick test")
        duration = time.time() - start

        assert duration < 0.1

    @pytest.mark.asyncio
    async def test_rrf_fusion_fast_execution(self):
        """Test RRF fusion completes within time limit."""
        import time
        reranker_mod = _get_reranker_module()
        SemanticReranker = reranker_mod.SemanticReranker
        Document = reranker_mod.Document

        reranker = SemanticReranker(enable_cache=False)

        # Create many documents
        lists = [
            [Document(id=f"{i}-{j}", content=f"Doc {i}-{j}") for j in range(100)]
            for i in range(5)
        ]

        start = time.time()
        await reranker.rrf_fusion(lists, k=60)
        duration = time.time() - start

        assert duration < 0.1


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
