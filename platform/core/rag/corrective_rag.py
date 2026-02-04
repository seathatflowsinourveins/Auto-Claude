"""
Corrective RAG (CRAG): Document Grading with Web Search Fallback

This module implements Corrective RAG, which evaluates retrieved documents
for relevance and falls back to web search when local retrieval fails.

Key Features:
- Document relevance grading (Correct/Ambiguous/Incorrect)
- Web search fallback for low-quality retrievals
- Knowledge refinement for ambiguous cases
- Configurable confidence thresholds

Architecture:
    Query -> Retriever -> Documents -> [Evaluator] -> Correct: Use context
                                          |         -> Ambiguous: Refine knowledge
                                          |         -> Incorrect: Web search
                                          v
                                      Generator -> Response

Reference: https://arxiv.org/abs/2401.15884

Integration:
    from core.rag.corrective_rag import CorrectiveRAG, CRAGConfig

    crag = CorrectiveRAG(
        llm=my_llm,
        retriever=my_retriever,
        web_search=tavily_search
    )
    result = await crag.generate("What are the latest AI trends?")
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...


class RetrieverProvider(Protocol):
    """Protocol for retriever providers."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for query."""
        ...


class WebSearchProvider(Protocol):
    """Protocol for web search providers (e.g., Tavily, Serper)."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search the web for query. Returns list of results with 'content' and 'url' keys."""
        ...


class GradingDecision(str, Enum):
    """Document grading decision."""
    CORRECT = "correct"          # Documents are relevant, use as context
    AMBIGUOUS = "ambiguous"      # Partial relevance, need refinement
    INCORRECT = "incorrect"      # Documents irrelevant, use web search


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CRAGConfig:
    """Configuration for Corrective RAG pipeline.

    Attributes:
        correct_threshold: Score above which docs are considered correct (default: 0.7)
        ambiguous_threshold: Score above which docs are ambiguous (default: 0.3)
        top_k: Number of documents to retrieve (default: 5)
        web_search_results: Number of web search results (default: 5)
        enable_knowledge_refinement: Refine ambiguous knowledge (default: True)
        enable_web_fallback: Fall back to web search (default: True)
        max_tokens: Maximum tokens for generation (default: 1024)
        temperature: Generation temperature (default: 0.7)
        rewrite_query_for_web: Rewrite query for web search (default: True)
    """
    correct_threshold: float = 0.7
    ambiguous_threshold: float = 0.3
    top_k: int = 5
    web_search_results: int = 5
    enable_knowledge_refinement: bool = True
    enable_web_fallback: bool = True
    max_tokens: int = 1024
    temperature: float = 0.7
    rewrite_query_for_web: bool = True


@dataclass
class GradedDocument:
    """A document with grading information."""
    content: str
    score: float
    grade: GradingDecision
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "retriever"  # "retriever" or "web_search"


@dataclass
class GradingResult:
    """Result of document grading phase."""
    decision: GradingDecision
    overall_score: float
    graded_documents: List[GradedDocument]
    correct_count: int
    ambiguous_count: int
    incorrect_count: int


@dataclass
class CRAGResult:
    """Final result from CRAG pipeline.

    Attributes:
        response: The final generated response
        grading_decision: Overall document grading decision
        source_type: Source of context ("retriever", "web_search", "mixed")
        documents_retrieved: Number of documents from retriever
        documents_from_web: Number of documents from web search
        relevant_documents: Total relevant documents used
        confidence: Confidence in the response
        graded_documents: List of graded documents
        web_query: Query used for web search (if any)
        refinement_applied: Whether knowledge refinement was applied
    """
    response: str
    grading_decision: GradingDecision
    source_type: str = "retriever"
    documents_retrieved: int = 0
    documents_from_web: int = 0
    relevant_documents: int = 0
    confidence: float = 0.0
    graded_documents: List[GradedDocument] = field(default_factory=list)
    web_query: Optional[str] = None
    refinement_applied: bool = False


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class CRAGPrompts:
    """Prompt templates for CRAG operations."""

    GRADE_DOCUMENT = """Evaluate if this document is relevant and useful for answering the query.

Query: {query}

Document:
{document}

Grading criteria:
- CORRECT: Document directly addresses the query with relevant, accurate information
- AMBIGUOUS: Document has some relevant information but is incomplete or tangentially related
- INCORRECT: Document is not relevant to the query

Provide your evaluation:
GRADE: [CORRECT/AMBIGUOUS/INCORRECT]
SCORE: [0.0-1.0] (confidence in relevance)
REASONING: [brief explanation]"""

    REFINE_KNOWLEDGE = """Extract and refine the relevant information from these documents for the query.

Query: {query}

Documents with partial relevance:
{documents}

Instructions:
1. Identify the key relevant facts from each document
2. Synthesize the information into a coherent summary
3. Note any gaps in the information

Refined knowledge:"""

    REWRITE_QUERY_FOR_WEB = """Rewrite this query for web search to find relevant, up-to-date information.

Original Query: {query}

Context: The local knowledge base did not have relevant information.

Instructions:
1. Make the query more specific and searchable
2. Include relevant keywords
3. Consider adding time-relevant terms if appropriate

Rewritten Query:"""

    GENERATE_WITH_CONTEXT = """Answer the query using the provided context.

Query: {query}

Context:
{context}

Instructions:
1. Use information from the context to answer
2. Be accurate and cite the context when relevant
3. If the context is insufficient, acknowledge limitations

Response:"""

    GENERATE_WITH_MIXED_CONTEXT = """Answer the query using information from both local knowledge and web search.

Query: {query}

Local Knowledge:
{local_context}

Web Search Results:
{web_context}

Instructions:
1. Prioritize accurate, well-supported information
2. Synthesize information from both sources
3. Note if sources conflict

Response:"""


# =============================================================================
# DOCUMENT GRADER
# =============================================================================

class DocumentGrader:
    """Grades documents for relevance using LLM evaluation."""

    def __init__(
        self,
        llm: LLMProvider,
        correct_threshold: float = 0.7,
        ambiguous_threshold: float = 0.3
    ):
        """Initialize document grader.

        Args:
            llm: LLM provider for grading
            correct_threshold: Score above which docs are CORRECT
            ambiguous_threshold: Score above which docs are AMBIGUOUS
        """
        self.llm = llm
        self.correct_threshold = correct_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.prompts = CRAGPrompts()

    async def grade_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> GradingResult:
        """Grade a list of documents for relevance.

        Args:
            query: The user query
            documents: List of documents to grade

        Returns:
            GradingResult with individual and overall grades
        """
        if not documents:
            return GradingResult(
                decision=GradingDecision.INCORRECT,
                overall_score=0.0,
                graded_documents=[],
                correct_count=0,
                ambiguous_count=0,
                incorrect_count=0
            )

        # Grade documents in parallel
        grading_tasks = [
            self._grade_single_document(query, doc)
            for doc in documents
        ]
        graded_documents = await asyncio.gather(*grading_tasks, return_exceptions=True)

        # Filter out exceptions and collect valid grades
        valid_grades: List[GradedDocument] = []
        for result in graded_documents:
            if isinstance(result, GradedDocument):
                valid_grades.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Document grading failed: {result}")

        # Calculate counts
        correct_count = sum(1 for g in valid_grades if g.grade == GradingDecision.CORRECT)
        ambiguous_count = sum(1 for g in valid_grades if g.grade == GradingDecision.AMBIGUOUS)
        incorrect_count = sum(1 for g in valid_grades if g.grade == GradingDecision.INCORRECT)

        # Calculate overall score
        if valid_grades:
            overall_score = sum(g.score for g in valid_grades) / len(valid_grades)
        else:
            overall_score = 0.0

        # Determine overall decision
        if correct_count > 0 and correct_count >= len(valid_grades) * 0.5:
            decision = GradingDecision.CORRECT
        elif correct_count + ambiguous_count >= len(valid_grades) * 0.5:
            decision = GradingDecision.AMBIGUOUS
        else:
            decision = GradingDecision.INCORRECT

        return GradingResult(
            decision=decision,
            overall_score=overall_score,
            graded_documents=valid_grades,
            correct_count=correct_count,
            ambiguous_count=ambiguous_count,
            incorrect_count=incorrect_count
        )

    async def _grade_single_document(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> GradedDocument:
        """Grade a single document."""
        content = document.get("content", str(document))
        metadata = document.get("metadata", {})

        prompt = self.prompts.GRADE_DOCUMENT.format(
            query=query,
            document=content[:3000]  # Truncate long documents
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.3
            )

            grade, score, reasoning = self._parse_grading_response(response)

            return GradedDocument(
                content=content,
                score=score,
                grade=grade,
                reasoning=reasoning,
                metadata=metadata,
                source="retriever"
            )
        except Exception as e:
            logger.warning(f"Failed to grade document: {e}")
            # Default to ambiguous on error
            return GradedDocument(
                content=content,
                score=0.5,
                grade=GradingDecision.AMBIGUOUS,
                reasoning=f"Grading error: {e}",
                metadata=metadata,
                source="retriever"
            )

    def _parse_grading_response(self, response: str) -> Tuple[GradingDecision, float, str]:
        """Parse the grading response from LLM."""
        response_upper = response.upper()

        # Parse grade
        if "GRADE: CORRECT" in response_upper or "GRADE:CORRECT" in response_upper:
            grade = GradingDecision.CORRECT
        elif "GRADE: AMBIGUOUS" in response_upper or "GRADE:AMBIGUOUS" in response_upper:
            grade = GradingDecision.AMBIGUOUS
        elif "GRADE: INCORRECT" in response_upper or "GRADE:INCORRECT" in response_upper:
            grade = GradingDecision.INCORRECT
        elif "CORRECT" in response_upper and "INCORRECT" not in response_upper:
            grade = GradingDecision.CORRECT
        elif "INCORRECT" in response_upper:
            grade = GradingDecision.INCORRECT
        else:
            grade = GradingDecision.AMBIGUOUS

        # Parse score
        score_match = re.search(r'SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = self._grade_to_default_score(grade)
        else:
            score = self._grade_to_default_score(grade)

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return grade, score, reasoning

    def _grade_to_default_score(self, grade: GradingDecision) -> float:
        """Get default score for a grade."""
        return {
            GradingDecision.CORRECT: 0.85,
            GradingDecision.AMBIGUOUS: 0.5,
            GradingDecision.INCORRECT: 0.15
        }.get(grade, 0.5)


# =============================================================================
# KNOWLEDGE REFINER
# =============================================================================

class KnowledgeRefiner:
    """Refines and extracts relevant knowledge from ambiguous documents."""

    def __init__(self, llm: LLMProvider):
        """Initialize knowledge refiner.

        Args:
            llm: LLM provider for refinement
        """
        self.llm = llm
        self.prompts = CRAGPrompts()

    async def refine(
        self,
        query: str,
        documents: List[GradedDocument]
    ) -> str:
        """Refine knowledge from partially relevant documents.

        Args:
            query: The user query
            documents: Documents to refine

        Returns:
            Refined knowledge string
        """
        if not documents:
            return ""

        # Prepare document content
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            grade_label = doc.grade.value.upper()
            doc_texts.append(f"[Document {i} - {grade_label} ({doc.score:.2f})]\n{doc.content}")

        prompt = self.prompts.REFINE_KNOWLEDGE.format(
            query=query,
            documents="\n\n".join(doc_texts)
        )

        try:
            refined = await self.llm.generate(
                prompt,
                max_tokens=800,
                temperature=0.5
            )
            return refined.strip()
        except Exception as e:
            logger.error(f"Knowledge refinement failed: {e}")
            # Fall back to concatenated content
            return "\n\n".join(doc.content for doc in documents)


# =============================================================================
# QUERY REWRITER
# =============================================================================

class QueryRewriter:
    """Rewrites queries for better web search results."""

    def __init__(self, llm: LLMProvider):
        """Initialize query rewriter.

        Args:
            llm: LLM provider for rewriting
        """
        self.llm = llm
        self.prompts = CRAGPrompts()

    async def rewrite_for_web(self, query: str) -> str:
        """Rewrite query for web search.

        Args:
            query: Original query

        Returns:
            Rewritten query optimized for web search
        """
        prompt = self.prompts.REWRITE_QUERY_FOR_WEB.format(query=query)

        try:
            rewritten = await self.llm.generate(
                prompt,
                max_tokens=100,
                temperature=0.5
            )
            return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original")
            return query


# =============================================================================
# CORRECTIVE RAG IMPLEMENTATION
# =============================================================================

class CorrectiveRAG:
    """
    Corrective Retrieval-Augmented Generation (CRAG).

    Implements the CRAG framework which evaluates retrieved documents and
    adaptively uses web search when local retrieval is insufficient.

    Features:
    - Document relevance grading
    - Web search fallback for poor retrievals
    - Knowledge refinement for ambiguous cases
    - Query rewriting for web search

    Example:
        >>> from core.rag.corrective_rag import CorrectiveRAG, CRAGConfig
        >>>
        >>> config = CRAGConfig(correct_threshold=0.7, enable_web_fallback=True)
        >>> crag = CorrectiveRAG(
        ...     llm=my_llm,
        ...     retriever=my_retriever,
        ...     web_search=tavily_adapter
        ... )
        >>>
        >>> result = await crag.generate("What are the latest AI developments?")
        >>> print(f"Response: {result.response}")
        >>> print(f"Source: {result.source_type}")
        >>> print(f"Grading: {result.grading_decision}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: RetrieverProvider,
        web_search: Optional[WebSearchProvider] = None,
        config: Optional[CRAGConfig] = None
    ):
        """Initialize Corrective RAG.

        Args:
            llm: LLM provider for generation and grading
            retriever: Retriever for local knowledge
            web_search: Optional web search provider
            config: Configuration options
        """
        self.llm = llm
        self.retriever = retriever
        self.web_search = web_search
        self.config = config or CRAGConfig()

        # Initialize components
        self.grader = DocumentGrader(
            llm=llm,
            correct_threshold=self.config.correct_threshold,
            ambiguous_threshold=self.config.ambiguous_threshold
        )
        self.refiner = KnowledgeRefiner(llm=llm)
        self.query_rewriter = QueryRewriter(llm=llm)
        self.prompts = CRAGPrompts()

    async def generate(
        self,
        query: str,
        **kwargs
    ) -> CRAGResult:
        """Generate a response using CRAG pipeline.

        Args:
            query: The user query
            **kwargs: Additional arguments passed to LLM

        Returns:
            CRAGResult with response and metadata
        """
        # Step 1: Retrieve documents
        try:
            raw_docs = await self.retriever.retrieve(query, top_k=self.config.top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raw_docs = []

        # Step 2: Grade documents
        grading_result = await self.grader.grade_documents(query, raw_docs)

        # Initialize result tracking
        graded_documents = grading_result.graded_documents
        web_docs: List[GradedDocument] = []
        web_query: Optional[str] = None
        refinement_applied = False

        # Step 3: Handle based on grading decision
        if grading_result.decision == GradingDecision.CORRECT:
            # Use retrieved documents as context
            context_docs = [
                d for d in graded_documents
                if d.grade in [GradingDecision.CORRECT, GradingDecision.AMBIGUOUS]
            ]
            context = "\n\n".join(d.content for d in context_docs)
            source_type = "retriever"

        elif grading_result.decision == GradingDecision.AMBIGUOUS:
            # Refine knowledge and optionally supplement with web search
            if self.config.enable_knowledge_refinement:
                context = await self.refiner.refine(query, graded_documents)
                refinement_applied = True
            else:
                context = "\n\n".join(d.content for d in graded_documents)

            # Supplement with web search if available
            if self.config.enable_web_fallback and self.web_search is not None:
                web_docs, web_query = await self._perform_web_search(query)
                if web_docs:
                    source_type = "mixed"
                else:
                    source_type = "retriever"
            else:
                source_type = "retriever"

        else:  # INCORRECT
            # Fall back to web search
            if self.config.enable_web_fallback and self.web_search is not None:
                web_docs, web_query = await self._perform_web_search(query)
                if web_docs:
                    context = "\n\n".join(d.content for d in web_docs)
                    source_type = "web_search"
                else:
                    # No web results either, use whatever we have
                    context = "\n\n".join(d.content for d in graded_documents) if graded_documents else ""
                    source_type = "retriever"
            else:
                # No web search available
                context = "\n\n".join(d.content for d in graded_documents) if graded_documents else ""
                source_type = "retriever"

        # Step 4: Generate response
        if source_type == "mixed" and web_docs:
            local_context = "\n\n".join(d.content for d in graded_documents[:3])
            web_context = "\n\n".join(d.content for d in web_docs)
            response = await self._generate_mixed(query, local_context, web_context, **kwargs)
        else:
            response = await self._generate_with_context(query, context, **kwargs)

        # Calculate confidence
        confidence = self._calculate_confidence(
            grading_result,
            source_type,
            len(web_docs)
        )

        # Count relevant documents
        relevant_count = sum(
            1 for d in graded_documents
            if d.grade in [GradingDecision.CORRECT, GradingDecision.AMBIGUOUS]
        )

        return CRAGResult(
            response=response,
            grading_decision=grading_result.decision,
            source_type=source_type,
            documents_retrieved=len(raw_docs),
            documents_from_web=len(web_docs),
            relevant_documents=relevant_count + len(web_docs),
            confidence=confidence,
            graded_documents=graded_documents + web_docs,
            web_query=web_query,
            refinement_applied=refinement_applied
        )

    async def _perform_web_search(
        self,
        query: str
    ) -> Tuple[List[GradedDocument], str]:
        """Perform web search with optional query rewriting.

        Args:
            query: Original query

        Returns:
            Tuple of (web documents, query used)
        """
        if self.web_search is None:
            return [], query

        # Optionally rewrite query
        if self.config.rewrite_query_for_web:
            search_query = await self.query_rewriter.rewrite_for_web(query)
        else:
            search_query = query

        try:
            results = await self.web_search.search(
                search_query,
                max_results=self.config.web_search_results
            )

            web_docs = []
            for result in results:
                content = result.get("content", result.get("snippet", str(result)))
                url = result.get("url", "")
                web_docs.append(GradedDocument(
                    content=content,
                    score=0.8,  # Assume web results are reasonably relevant
                    grade=GradingDecision.CORRECT,
                    metadata={"url": url},
                    source="web_search"
                ))

            return web_docs, search_query

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [], search_query

    async def _generate_with_context(
        self,
        query: str,
        context: str,
        **kwargs
    ) -> str:
        """Generate response with single context source."""
        if not context:
            context = "No relevant context available."

        prompt = self.prompts.GENERATE_WITH_CONTEXT.format(
            query=query,
            context=context
        )

        response = await self.llm.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
        )

        return response.strip()

    async def _generate_mixed(
        self,
        query: str,
        local_context: str,
        web_context: str,
        **kwargs
    ) -> str:
        """Generate response with mixed context sources."""
        prompt = self.prompts.GENERATE_WITH_MIXED_CONTEXT.format(
            query=query,
            local_context=local_context or "No local knowledge available.",
            web_context=web_context or "No web results available."
        )

        response = await self.llm.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        return response.strip()

    def _calculate_confidence(
        self,
        grading_result: GradingResult,
        source_type: str,
        web_doc_count: int
    ) -> float:
        """Calculate confidence in the response."""
        base_confidence = grading_result.overall_score

        # Adjust based on source type
        if source_type == "retriever" and grading_result.decision == GradingDecision.CORRECT:
            confidence = base_confidence * 1.1  # Boost for good local retrieval
        elif source_type == "web_search":
            confidence = 0.7 + (0.3 * (web_doc_count / self.config.web_search_results))
        elif source_type == "mixed":
            confidence = (base_confidence + 0.7) / 2  # Average local and web confidence
        else:
            confidence = base_confidence

        return max(0.0, min(1.0, confidence))


# =============================================================================
# INTEGRATION WITH SEMANTIC RERANKER
# =============================================================================

class CRAGWithReranker:
    """
    CRAG integrated with SemanticReranker for improved document selection.

    Example:
        >>> from core.rag.corrective_rag import CRAGWithReranker
        >>> from core.rag.reranker import SemanticReranker
        >>>
        >>> reranker = SemanticReranker()
        >>> crag = CRAGWithReranker(
        ...     llm=my_llm,
        ...     retriever=my_retriever,
        ...     reranker=reranker,
        ...     web_search=tavily_search
        ... )
        >>> result = await crag.generate("What is RAG?")
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: RetrieverProvider,
        reranker: Any,  # SemanticReranker
        web_search: Optional[WebSearchProvider] = None,
        config: Optional[CRAGConfig] = None
    ):
        """Initialize CRAG with reranker.

        Args:
            llm: LLM provider
            retriever: Retriever provider
            reranker: SemanticReranker instance
            web_search: Optional web search provider
            config: Configuration
        """
        self.reranker = reranker
        self._wrapped_retriever = _RerankedRetriever(retriever, reranker)
        self.crag = CorrectiveRAG(
            llm=llm,
            retriever=self._wrapped_retriever,
            web_search=web_search,
            config=config
        )

    async def generate(self, query: str, **kwargs) -> CRAGResult:
        """Generate using CRAG with reranking."""
        return await self.crag.generate(query, **kwargs)


class _RerankedRetriever:
    """Internal wrapper that applies reranking to retrieved documents."""

    def __init__(self, base_retriever: RetrieverProvider, reranker: Any):
        self.base_retriever = base_retriever
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank documents."""
        # Get more documents for reranking
        raw_docs = await self.base_retriever.retrieve(query, top_k=top_k * 2)

        if not raw_docs:
            return []

        try:
            from .reranker import Document

            documents = [
                Document(
                    id=str(i),
                    content=doc.get("content", str(doc)),
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(raw_docs)
            ]

            reranked = await self.reranker.rerank(query, documents, top_k=top_k)

            return [
                {
                    "content": sd.document.content,
                    "metadata": sd.document.metadata,
                    "score": sd.score
                }
                for sd in reranked
            ]
        except ImportError:
            return raw_docs[:top_k]


# =============================================================================
# TAVILY ADAPTER EXAMPLE
# =============================================================================

class TavilySearchAdapter:
    """
    Example adapter for Tavily web search API.

    Note: Requires tavily-python package and TAVILY_API_KEY environment variable.

    Example:
        >>> adapter = TavilySearchAdapter()
        >>> results = await adapter.search("latest AI news")
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily adapter.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy load Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                import os

                api_key = self.api_key or os.environ.get("TAVILY_API_KEY")
                if not api_key:
                    raise ValueError("TAVILY_API_KEY not set")

                self._client = TavilyClient(api_key=api_key)
            except ImportError:
                raise ImportError("tavily-python not installed. Install with: pip install tavily-python")

        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of search results
        """
        # Run sync Tavily client in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._search_sync(query, max_results)
        )
        return results

    def _search_sync(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Synchronous search implementation."""
        client = self._get_client()
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "score": item.get("score", 0.0)
            })

        return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "CorrectiveRAG",
    "CRAGWithReranker",
    # Configuration
    "CRAGConfig",
    # Result types
    "CRAGResult",
    "GradedDocument",
    "GradingResult",
    # Components
    "DocumentGrader",
    "KnowledgeRefiner",
    "QueryRewriter",
    # Enums
    "GradingDecision",
    # Adapters
    "TavilySearchAdapter",
    # Protocols
    "LLMProvider",
    "RetrieverProvider",
    "WebSearchProvider",
    # Prompts
    "CRAGPrompts",
]
