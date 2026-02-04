"""
Self-RAG: Self-Reflective Retrieval-Augmented Generation

This module implements Self-RAG, a framework that enables LLMs to adaptively
retrieve, generate, and critique their own outputs through reflection tokens.

Key Features:
- Adaptive retrieval based on query complexity
- Reflection tokens for self-critique ([Retrieve], [IsREL], [IsSUP], [IsUSE])
- Iterative refinement loop with configurable iterations
- Support for custom LLM providers

Architecture:
    Query -> [Retrieve?] -> Retrieve (if yes) -> [Relevant?] -> Generate
                                                      |
                                                 [Supported?] -> Output/Regenerate

Reference: https://arxiv.org/abs/2310.11511

Integration:
    from core.rag.self_rag import SelfRAG, SelfRAGConfig, ReflectionResult

    self_rag = SelfRAG(llm=my_llm, retriever=my_retriever)
    result = await self_rag.generate("What are the benefits of microservices?")

    print(f"Response: {result.response}")
    print(f"Retrieval used: {result.retrieval_used}")
    print(f"Confidence: {result.confidence}")
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
        """Retrieve documents for query. Returns list of dicts with 'content' key."""
        ...


class ReflectionToken(str, Enum):
    """Self-RAG reflection tokens for decision making."""
    RETRIEVE = "[Retrieve]"      # Should we retrieve?
    IS_RELEVANT = "[IsREL]"      # Is retrieved doc relevant?
    IS_SUPPORTED = "[IsSUP]"     # Is response supported by evidence?
    IS_USEFUL = "[IsUSE]"        # Is response useful for the query?


class RetrievalDecision(str, Enum):
    """Decision on whether to retrieve."""
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class RelevanceGrade(str, Enum):
    """Relevance grade for documents."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


class SupportGrade(str, Enum):
    """Support grade for generated content."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class UsefulnessGrade(str, Enum):
    """Usefulness grade for response."""
    VERY_USEFUL = "5"
    USEFUL = "4"
    SOMEWHAT_USEFUL = "3"
    SLIGHTLY_USEFUL = "2"
    NOT_USEFUL = "1"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SelfRAGConfig:
    """Configuration for Self-RAG pipeline.

    Attributes:
        max_iterations: Maximum refinement iterations (default: 3)
        retrieval_threshold: Confidence threshold to skip retrieval (default: 0.7)
        relevance_threshold: Minimum relevance score for documents (default: 0.5)
        support_threshold: Minimum support score for output (default: 0.6)
        usefulness_threshold: Minimum usefulness score (default: 3)
        top_k: Number of documents to retrieve (default: 5)
        enable_adaptive_retrieval: Whether to adaptively decide to retrieve (default: True)
        enable_iterative_refinement: Whether to refine iteratively (default: True)
        max_tokens: Maximum tokens for generation (default: 1024)
        temperature: Generation temperature (default: 0.7)
    """
    max_iterations: int = 3
    retrieval_threshold: float = 0.7
    relevance_threshold: float = 0.5
    support_threshold: float = 0.6
    usefulness_threshold: int = 3
    top_k: int = 5
    enable_adaptive_retrieval: bool = True
    enable_iterative_refinement: bool = True
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class RetrievedDocument:
    """A retrieved document with relevance score."""
    content: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_relevant: bool = False


@dataclass
class ReflectionResult:
    """Result of a reflection evaluation."""
    token: ReflectionToken
    decision: str
    confidence: float
    reasoning: str = ""


@dataclass
class GenerationAttempt:
    """A single generation attempt with metadata."""
    response: str
    context_used: List[str]
    support_score: float
    usefulness_score: int
    iteration: int


@dataclass
class SelfRAGResult:
    """Final result from Self-RAG pipeline.

    Attributes:
        response: The final generated response
        retrieval_used: Whether retrieval was used
        documents_retrieved: Number of documents retrieved
        relevant_documents: Number of relevant documents used
        iterations: Number of refinement iterations
        confidence: Overall confidence in the response
        support_grade: How well the response is supported by evidence
        usefulness_grade: How useful the response is
        generation_attempts: History of generation attempts
        reflection_trace: Trace of all reflection decisions
    """
    response: str
    retrieval_used: bool = False
    documents_retrieved: int = 0
    relevant_documents: int = 0
    iterations: int = 1
    confidence: float = 0.0
    support_grade: SupportGrade = SupportGrade.NOT_SUPPORTED
    usefulness_grade: UsefulnessGrade = UsefulnessGrade.NOT_USEFUL
    generation_attempts: List[GenerationAttempt] = field(default_factory=list)
    reflection_trace: List[ReflectionResult] = field(default_factory=list)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class SelfRAGPrompts:
    """Prompt templates for Self-RAG operations."""

    RETRIEVAL_DECISION = """Analyze this query and decide if external retrieval is needed.

Query: {query}

Consider:
1. Is this a factual question requiring specific knowledge?
2. Does this require recent or specialized information?
3. Can this be answered from general knowledge?

Respond with ONLY one of: YES, NO, or UNCERTAIN
Then provide a brief reasoning.

Format:
DECISION: [YES/NO/UNCERTAIN]
REASONING: [your reasoning]"""

    RELEVANCE_CHECK = """Evaluate if this document is relevant to the query.

Query: {query}

Document:
{document}

Rate the relevance:
- RELEVANT: Directly answers or contains information for the query
- PARTIALLY_RELEVANT: Contains some useful information
- IRRELEVANT: Not useful for answering the query

Respond with:
RELEVANCE: [RELEVANT/PARTIALLY_RELEVANT/IRRELEVANT]
SCORE: [0.0-1.0]
REASONING: [brief explanation]"""

    GENERATE_WITH_CONTEXT = """Answer the query using the provided context.

Query: {query}

Context:
{context}

Instructions:
1. Use only information from the context
2. If context is insufficient, state what's missing
3. Be specific and cite relevant parts of context

Response:"""

    GENERATE_WITHOUT_CONTEXT = """Answer the query using your knowledge.

Query: {query}

Instructions:
1. Provide a helpful, accurate response
2. If uncertain, express your confidence level
3. Be specific and informative

Response:"""

    SUPPORT_CHECK = """Evaluate if this response is supported by the given context.

Response:
{response}

Context:
{context}

Evaluate:
- FULLY_SUPPORTED: All claims in response are backed by context
- PARTIALLY_SUPPORTED: Some claims are backed, others are not
- NOT_SUPPORTED: Response makes claims not in context

Respond with:
SUPPORT: [FULLY_SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED]
SCORE: [0.0-1.0]
UNSUPPORTED_CLAIMS: [list any claims not in context]"""

    USEFULNESS_CHECK = """Rate how useful this response is for the query.

Query: {query}

Response:
{response}

Rate from 1-5:
5 = Very useful, directly and completely answers the query
4 = Useful, mostly answers the query
3 = Somewhat useful, partial answer
2 = Slightly useful, tangentially related
1 = Not useful, doesn't address the query

Respond with:
USEFULNESS: [1-5]
REASONING: [brief explanation]"""

    REFINE_RESPONSE = """Improve this response based on the feedback.

Original Query: {query}

Original Response:
{response}

Issues Identified:
{issues}

Context Available:
{context}

Generate an improved response that addresses the identified issues:"""


# =============================================================================
# SELF-RAG IMPLEMENTATION
# =============================================================================

class SelfRAG:
    """
    Self-Reflective Retrieval-Augmented Generation.

    Implements the Self-RAG framework which teaches LLMs to adaptively retrieve,
    generate, and critique using reflection tokens.

    Features:
    - Adaptive retrieval decision based on query complexity
    - Document relevance grading
    - Response support verification against context
    - Usefulness evaluation
    - Iterative refinement loop

    Example:
        >>> from core.rag.self_rag import SelfRAG, SelfRAGConfig
        >>>
        >>> config = SelfRAGConfig(max_iterations=3, top_k=5)
        >>> self_rag = SelfRAG(llm=my_llm, retriever=my_retriever, config=config)
        >>>
        >>> result = await self_rag.generate("What is microservices architecture?")
        >>> print(f"Response: {result.response}")
        >>> print(f"Confidence: {result.confidence}")
        >>> print(f"Support: {result.support_grade}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Optional[RetrieverProvider] = None,
        config: Optional[SelfRAGConfig] = None,
    ):
        """Initialize Self-RAG.

        Args:
            llm: LLM provider for generation and reflection
            retriever: Optional retriever for document retrieval
            config: Configuration options
        """
        self.llm = llm
        self.retriever = retriever
        self.config = config or SelfRAGConfig()
        self.prompts = SelfRAGPrompts()

    async def generate(
        self,
        query: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> SelfRAGResult:
        """Generate a response using Self-RAG pipeline.

        Args:
            query: The user query
            context: Optional pre-provided context
            **kwargs: Additional arguments passed to LLM

        Returns:
            SelfRAGResult with response and metadata
        """
        reflection_trace: List[ReflectionResult] = []
        generation_attempts: List[GenerationAttempt] = []

        # Step 1: Decide whether to retrieve
        retrieval_decision = await self._decide_retrieval(query)
        reflection_trace.append(retrieval_decision)

        retrieval_used = False
        documents: List[RetrievedDocument] = []

        if context:
            # Use provided context
            documents = [
                RetrievedDocument(content=c, relevance_score=1.0, is_relevant=True)
                for c in context
            ]
            retrieval_used = True
        elif (
            self.config.enable_adaptive_retrieval and
            retrieval_decision.decision in [RetrievalDecision.YES.value, RetrievalDecision.UNCERTAIN.value] and
            self.retriever is not None
        ):
            # Retrieve documents
            documents = await self._retrieve_and_grade(query)
            retrieval_used = True

        # Filter to relevant documents
        relevant_docs = [d for d in documents if d.is_relevant]
        context_texts = [d.content for d in relevant_docs]

        # Step 2: Generate response
        response = await self._generate_response(query, context_texts, **kwargs)

        # Step 3: Evaluate support
        support_result = await self._check_support(response, context_texts)
        reflection_trace.append(support_result)

        # Step 4: Evaluate usefulness
        usefulness_result = await self._check_usefulness(query, response)
        reflection_trace.append(usefulness_result)

        # Record first attempt
        generation_attempts.append(GenerationAttempt(
            response=response,
            context_used=context_texts,
            support_score=support_result.confidence,
            usefulness_score=int(usefulness_result.decision),
            iteration=1
        ))

        # Step 5: Iterative refinement if needed
        iteration = 1
        if self.config.enable_iterative_refinement:
            while iteration < self.config.max_iterations:
                # Check if refinement is needed
                needs_refinement = (
                    support_result.confidence < self.config.support_threshold or
                    int(usefulness_result.decision) < self.config.usefulness_threshold
                )

                if not needs_refinement:
                    break

                iteration += 1

                # Identify issues
                issues = self._identify_issues(support_result, usefulness_result)

                # Refine response
                response = await self._refine_response(
                    query, response, issues, context_texts, **kwargs
                )

                # Re-evaluate
                support_result = await self._check_support(response, context_texts)
                reflection_trace.append(support_result)

                usefulness_result = await self._check_usefulness(query, response)
                reflection_trace.append(usefulness_result)

                generation_attempts.append(GenerationAttempt(
                    response=response,
                    context_used=context_texts,
                    support_score=support_result.confidence,
                    usefulness_score=int(usefulness_result.decision),
                    iteration=iteration
                ))

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            support_result.confidence,
            int(usefulness_result.decision),
            len(relevant_docs),
            retrieval_used
        )

        return SelfRAGResult(
            response=response,
            retrieval_used=retrieval_used,
            documents_retrieved=len(documents),
            relevant_documents=len(relevant_docs),
            iterations=iteration,
            confidence=confidence,
            support_grade=self._parse_support_grade(support_result.decision),
            usefulness_grade=self._parse_usefulness_grade(usefulness_result.decision),
            generation_attempts=generation_attempts,
            reflection_trace=reflection_trace
        )

    async def _decide_retrieval(self, query: str) -> ReflectionResult:
        """Decide whether retrieval is needed for this query."""
        if not self.config.enable_adaptive_retrieval or self.retriever is None:
            return ReflectionResult(
                token=ReflectionToken.RETRIEVE,
                decision=RetrievalDecision.NO.value,
                confidence=1.0,
                reasoning="Retrieval disabled or no retriever available"
            )

        prompt = self.prompts.RETRIEVAL_DECISION.format(query=query)

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=150,
                temperature=0.3
            )

            decision, reasoning = self._parse_retrieval_response(response)

            return ReflectionResult(
                token=ReflectionToken.RETRIEVE,
                decision=decision,
                confidence=0.8 if decision == RetrievalDecision.YES.value else 0.7,
                reasoning=reasoning
            )
        except Exception as e:
            logger.warning(f"Retrieval decision failed: {e}, defaulting to YES")
            return ReflectionResult(
                token=ReflectionToken.RETRIEVE,
                decision=RetrievalDecision.YES.value,
                confidence=0.5,
                reasoning=f"Default due to error: {e}"
            )

    def _parse_retrieval_response(self, response: str) -> Tuple[str, str]:
        """Parse the retrieval decision response."""
        response_upper = response.upper()

        if "DECISION: YES" in response_upper or "DECISION:YES" in response_upper:
            decision = RetrievalDecision.YES.value
        elif "DECISION: NO" in response_upper or "DECISION:NO" in response_upper:
            decision = RetrievalDecision.NO.value
        elif "YES" in response_upper:
            decision = RetrievalDecision.YES.value
        elif "NO" in response_upper:
            decision = RetrievalDecision.NO.value
        else:
            decision = RetrievalDecision.UNCERTAIN.value

        # Extract reasoning
        reasoning = ""
        if "REASONING:" in response.upper():
            reasoning = response.split("REASONING:", 1)[-1].strip()
        elif "\n" in response:
            reasoning = response.split("\n", 1)[-1].strip()

        return decision, reasoning

    async def _retrieve_and_grade(self, query: str) -> List[RetrievedDocument]:
        """Retrieve documents and grade their relevance."""
        if self.retriever is None:
            return []

        try:
            raw_docs = await self.retriever.retrieve(query, top_k=self.config.top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

        documents: List[RetrievedDocument] = []

        for doc in raw_docs:
            content = doc.get("content", str(doc))
            metadata = doc.get("metadata", {})

            # Grade relevance
            relevance_result = await self._check_relevance(query, content)

            is_relevant = relevance_result.confidence >= self.config.relevance_threshold

            documents.append(RetrievedDocument(
                content=content,
                relevance_score=relevance_result.confidence,
                metadata=metadata,
                is_relevant=is_relevant
            ))

        return documents

    async def _check_relevance(self, query: str, document: str) -> ReflectionResult:
        """Check if a document is relevant to the query."""
        prompt = self.prompts.RELEVANCE_CHECK.format(
            query=query,
            document=document[:2000]  # Truncate long documents
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.3
            )

            grade, score, reasoning = self._parse_relevance_response(response)

            return ReflectionResult(
                token=ReflectionToken.IS_RELEVANT,
                decision=grade,
                confidence=score,
                reasoning=reasoning
            )
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}")
            return ReflectionResult(
                token=ReflectionToken.IS_RELEVANT,
                decision=RelevanceGrade.PARTIALLY_RELEVANT.value,
                confidence=0.5,
                reasoning=f"Error: {e}"
            )

    def _parse_relevance_response(self, response: str) -> Tuple[str, float, str]:
        """Parse relevance check response."""
        response_upper = response.upper()

        # Parse grade
        if "RELEVANT" in response_upper and "IRRELEVANT" not in response_upper:
            if "PARTIALLY" in response_upper:
                grade = RelevanceGrade.PARTIALLY_RELEVANT.value
            else:
                grade = RelevanceGrade.RELEVANT.value
        elif "IRRELEVANT" in response_upper:
            grade = RelevanceGrade.IRRELEVANT.value
        else:
            grade = RelevanceGrade.PARTIALLY_RELEVANT.value

        # Parse score
        score = 0.5
        score_match = re.search(r'SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif grade == RelevanceGrade.RELEVANT.value:
            score = 0.9
        elif grade == RelevanceGrade.IRRELEVANT.value:
            score = 0.1

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return grade, score, reasoning

    async def _generate_response(
        self,
        query: str,
        context: List[str],
        **kwargs
    ) -> str:
        """Generate response with or without context."""
        if context:
            prompt = self.prompts.GENERATE_WITH_CONTEXT.format(
                query=query,
                context="\n\n".join(context)
            )
        else:
            prompt = self.prompts.GENERATE_WITHOUT_CONTEXT.format(query=query)

        response = await self.llm.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
        )

        return response.strip()

    async def _check_support(
        self,
        response: str,
        context: List[str]
    ) -> ReflectionResult:
        """Check if response is supported by context."""
        if not context:
            # No context to check against
            return ReflectionResult(
                token=ReflectionToken.IS_SUPPORTED,
                decision=SupportGrade.NOT_SUPPORTED.value,
                confidence=0.0,
                reasoning="No context provided"
            )

        prompt = self.prompts.SUPPORT_CHECK.format(
            response=response,
            context="\n\n".join(context)
        )

        try:
            llm_response = await self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.3
            )

            grade, score, unsupported = self._parse_support_response(llm_response)

            return ReflectionResult(
                token=ReflectionToken.IS_SUPPORTED,
                decision=grade,
                confidence=score,
                reasoning=f"Unsupported claims: {unsupported}" if unsupported else ""
            )
        except Exception as e:
            logger.warning(f"Support check failed: {e}")
            return ReflectionResult(
                token=ReflectionToken.IS_SUPPORTED,
                decision=SupportGrade.PARTIALLY_SUPPORTED.value,
                confidence=0.5,
                reasoning=f"Error: {e}"
            )

    def _parse_support_response(self, response: str) -> Tuple[str, float, str]:
        """Parse support check response."""
        response_upper = response.upper()

        # Parse grade
        if "FULLY_SUPPORTED" in response_upper or "FULLY SUPPORTED" in response_upper:
            grade = SupportGrade.FULLY_SUPPORTED.value
            default_score = 0.95
        elif "PARTIALLY_SUPPORTED" in response_upper or "PARTIALLY SUPPORTED" in response_upper:
            grade = SupportGrade.PARTIALLY_SUPPORTED.value
            default_score = 0.6
        elif "NOT_SUPPORTED" in response_upper or "NOT SUPPORTED" in response_upper:
            grade = SupportGrade.NOT_SUPPORTED.value
            default_score = 0.2
        else:
            grade = SupportGrade.PARTIALLY_SUPPORTED.value
            default_score = 0.5

        # Parse score
        score = default_score
        score_match = re.search(r'SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Parse unsupported claims
        unsupported = ""
        unsupported_match = re.search(
            r'UNSUPPORTED_CLAIMS:\s*(.+?)(?:\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if unsupported_match:
            unsupported = unsupported_match.group(1).strip()

        return grade, score, unsupported

    async def _check_usefulness(self, query: str, response: str) -> ReflectionResult:
        """Check usefulness of response for the query."""
        prompt = self.prompts.USEFULNESS_CHECK.format(
            query=query,
            response=response
        )

        try:
            llm_response = await self.llm.generate(
                prompt,
                max_tokens=150,
                temperature=0.3
            )

            score, reasoning = self._parse_usefulness_response(llm_response)

            return ReflectionResult(
                token=ReflectionToken.IS_USEFUL,
                decision=str(score),
                confidence=score / 5.0,
                reasoning=reasoning
            )
        except Exception as e:
            logger.warning(f"Usefulness check failed: {e}")
            return ReflectionResult(
                token=ReflectionToken.IS_USEFUL,
                decision="3",
                confidence=0.6,
                reasoning=f"Error: {e}"
            )

    def _parse_usefulness_response(self, response: str) -> Tuple[int, str]:
        """Parse usefulness check response."""
        # Find usefulness score
        score = 3
        score_match = re.search(r'USEFULNESS:\s*(\d)', response, re.IGNORECASE)
        if score_match:
            try:
                score = int(score_match.group(1))
                score = max(1, min(5, score))
            except ValueError:
                pass
        else:
            # Try to find any digit 1-5
            digit_match = re.search(r'\b([1-5])\b', response)
            if digit_match:
                score = int(digit_match.group(1))

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning

    def _identify_issues(
        self,
        support_result: ReflectionResult,
        usefulness_result: ReflectionResult
    ) -> str:
        """Identify issues for refinement."""
        issues = []

        if support_result.confidence < self.config.support_threshold:
            issues.append(f"Support issue: {support_result.reasoning or 'Response not well supported by context'}")

        if int(usefulness_result.decision) < self.config.usefulness_threshold:
            issues.append(f"Usefulness issue: {usefulness_result.reasoning or 'Response not sufficiently useful'}")

        return "\n".join(issues) if issues else "Improve overall quality and accuracy"

    async def _refine_response(
        self,
        query: str,
        response: str,
        issues: str,
        context: List[str],
        **kwargs
    ) -> str:
        """Refine response based on identified issues."""
        prompt = self.prompts.REFINE_RESPONSE.format(
            query=query,
            response=response,
            issues=issues,
            context="\n\n".join(context) if context else "No additional context available"
        )

        refined = await self.llm.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature * 0.8),  # Lower temp for refinement
        )

        return refined.strip()

    def _calculate_confidence(
        self,
        support_score: float,
        usefulness_score: int,
        relevant_docs: int,
        retrieval_used: bool
    ) -> float:
        """Calculate overall confidence score."""
        # Normalize usefulness to 0-1
        usefulness_normalized = usefulness_score / 5.0

        # Weight components
        if retrieval_used and relevant_docs > 0:
            # With retrieval: weight support heavily
            confidence = (
                support_score * 0.5 +
                usefulness_normalized * 0.3 +
                min(relevant_docs / 3.0, 1.0) * 0.2
            )
        else:
            # Without retrieval: weight usefulness more
            confidence = (
                support_score * 0.3 +
                usefulness_normalized * 0.7
            )

        return max(0.0, min(1.0, confidence))

    def _parse_support_grade(self, decision: str) -> SupportGrade:
        """Convert decision string to SupportGrade enum."""
        try:
            return SupportGrade(decision)
        except ValueError:
            return SupportGrade.PARTIALLY_SUPPORTED

    def _parse_usefulness_grade(self, decision: str) -> UsefulnessGrade:
        """Convert decision string to UsefulnessGrade enum."""
        try:
            return UsefulnessGrade(decision)
        except ValueError:
            return UsefulnessGrade.SOMEWHAT_USEFUL


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

class SelfRAGWithReranker:
    """
    Self-RAG integrated with SemanticReranker for improved retrieval quality.

    This wrapper combines Self-RAG's adaptive retrieval with semantic reranking
    for better document selection.

    Example:
        >>> from core.rag.self_rag import SelfRAGWithReranker
        >>> from core.rag.reranker import SemanticReranker, Document
        >>>
        >>> reranker = SemanticReranker()
        >>> self_rag = SelfRAGWithReranker(
        ...     llm=my_llm,
        ...     retriever=my_retriever,
        ...     reranker=reranker
        ... )
        >>> result = await self_rag.generate("What is RAG?")
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: RetrieverProvider,
        reranker: Any,  # SemanticReranker
        config: Optional[SelfRAGConfig] = None,
    ):
        """Initialize Self-RAG with reranker.

        Args:
            llm: LLM provider
            retriever: Retriever provider
            reranker: SemanticReranker instance
            config: Configuration
        """
        self.reranker = reranker
        self._wrapped_retriever = _RerankedRetriever(retriever, reranker)
        self.self_rag = SelfRAG(
            llm=llm,
            retriever=self._wrapped_retriever,
            config=config
        )

    async def generate(
        self,
        query: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> SelfRAGResult:
        """Generate using Self-RAG with reranking."""
        return await self.self_rag.generate(query, context, **kwargs)


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

        # Convert to Document objects for reranker
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

            # Rerank
            reranked = await self.reranker.rerank(query, documents, top_k=top_k)

            # Convert back to dict format
            return [
                {
                    "content": sd.document.content,
                    "metadata": sd.document.metadata,
                    "score": sd.score
                }
                for sd in reranked
            ]
        except ImportError:
            # Fallback if reranker types not available
            return raw_docs[:top_k]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "SelfRAG",
    "SelfRAGWithReranker",
    # Configuration
    "SelfRAGConfig",
    # Result types
    "SelfRAGResult",
    "ReflectionResult",
    "RetrievedDocument",
    "GenerationAttempt",
    # Enums
    "ReflectionToken",
    "RetrievalDecision",
    "RelevanceGrade",
    "SupportGrade",
    "UsefulnessGrade",
    # Protocols
    "LLMProvider",
    "RetrieverProvider",
    # Prompts
    "SelfRAGPrompts",
]
