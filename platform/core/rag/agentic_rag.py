"""
Agentic RAG Loop: Iterative Retrieval-Generation with Tool Augmentation

This module implements an agentic RAG loop that orchestrates iterative retrieval
and generation with dynamic tool selection, query decomposition, and self-critique.

Key Features:
- Iterative retrieval-generation loop with state machine
- Tool-augmented retrieval (Exa, Tavily, Context7, memory search)
- Query decomposition for complex queries
- Reflection and self-critique for quality assurance
- Integration with Self-RAG, CRAG, HyDE, RAPTOR
- RRF fusion for combining multi-source results
- Async execution for performance

Architecture:
    State Machine: PLAN -> RETRIEVE -> GENERATE -> EVALUATE -> (REFINE | COMPLETE)

Reference: https://arxiv.org/abs/2312.10997 (Adaptive-RAG)

Integration:
    from core.rag.agentic_rag import AgenticRAG, AgenticRAGConfig

    agentic_rag = AgenticRAG(
        llm=my_llm,
        tools=[exa_tool, tavily_tool, memory_tool],
        config=AgenticRAGConfig(max_iterations=5, confidence_threshold=0.8)
    )
    result = await agentic_rag.run("Complex multi-part question about AI systems")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    TypeVar,
    Set,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def encode(self, texts: Union[str, List[str]]) -> Any:
        """Encode text(s) to embedding(s)."""
        ...


class RetrievalTool(Protocol):
    """Protocol for retrieval tools."""

    @property
    def name(self) -> str:
        """Tool name for identification."""
        ...

    @property
    def description(self) -> str:
        """Tool description for LLM tool selection."""
        ...

    @property
    def supported_query_types(self) -> List[str]:
        """Query types this tool handles well."""
        ...

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve documents. Returns list with 'content', 'score', 'metadata'."""
        ...


class RAGStrategy(Protocol):
    """Protocol for RAG strategies (Self-RAG, CRAG, HyDE, RAPTOR)."""

    async def generate(
        self,
        query: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """Generate response using the strategy."""
        ...


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AgentState(str, Enum):
    """States in the agentic RAG state machine."""
    INIT = "init"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    EVALUATE = "evaluate"
    REFINE = "refine"
    COMPLETE = "complete"
    ERROR = "error"


class QueryType(str, Enum):
    """Types of queries for tool selection."""
    FACTUAL = "factual"              # Simple fact lookup
    RESEARCH = "research"            # Deep research needed
    CODE = "code"                    # Code-related
    NEWS = "news"                    # Current events
    MULTI_HOP = "multi_hop"          # Requires multiple reasoning steps
    COMPARISON = "comparison"        # Compare/contrast
    EXPLANATION = "explanation"      # Explain concept
    GENERAL = "general"              # General knowledge


class EvaluationDecision(str, Enum):
    """Decisions from evaluation phase."""
    COMPLETE = "complete"            # Response is satisfactory
    REFINE = "refine"                # Need to refine/improve
    RETRY_RETRIEVE = "retry_retrieve"  # Need more/better context
    DECOMPOSE = "decompose"          # Query needs decomposition
    FAIL = "fail"                    # Cannot generate satisfactory response


@dataclass
class AgenticRAGConfig:
    """Configuration for Agentic RAG loop.

    Attributes:
        max_iterations: Maximum retrieval-generation iterations (default: 5)
        confidence_threshold: Early stopping threshold (default: 0.8)
        enable_query_decomposition: Decompose complex queries (default: True)
        enable_reflection: Enable self-critique (default: True)
        enable_tool_selection: Dynamic tool selection (default: True)
        top_k_per_source: Documents per retrieval source (default: 5)
        max_total_documents: Maximum total documents (default: 20)
        generation_max_tokens: Maximum tokens for generation (default: 2048)
        temperature: Generation temperature (default: 0.7)
        relevance_threshold: Minimum relevance for documents (default: 0.5)
        completeness_threshold: Minimum completeness score (default: 0.6)
        fusion_method: How to combine results - 'rrf', 'weighted', 'interleave' (default: 'rrf')
        rrf_k: RRF constant (default: 60)
        enable_caching: Cache intermediate results (default: True)
        timeout_seconds: Overall timeout (default: 120)
    """
    max_iterations: int = 5
    confidence_threshold: float = 0.8
    enable_query_decomposition: bool = True
    enable_reflection: bool = True
    enable_tool_selection: bool = True
    top_k_per_source: int = 5
    max_total_documents: int = 20
    generation_max_tokens: int = 2048
    temperature: float = 0.7
    relevance_threshold: float = 0.5
    completeness_threshold: float = 0.6
    fusion_method: str = "rrf"
    rrf_k: int = 60
    enable_caching: bool = True
    timeout_seconds: float = 120.0


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str
    query_type: QueryType
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class RetrievedContext:
    """Retrieved context from a single source."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""


@dataclass
class RetrievalResult:
    """Combined result from retrieval phase."""
    contexts: List[RetrievedContext]
    total_retrieved: int
    sources_used: List[str]
    latency_ms: float


@dataclass
class EvaluationResult:
    """Result from evaluation phase."""
    decision: EvaluationDecision
    confidence: float
    relevance_score: float
    completeness_score: float
    issues: List[str]
    suggestions: List[str]


@dataclass
class GenerationResult:
    """Result from a single generation attempt."""
    response: str
    iteration: int
    contexts_used: int
    confidence: float
    reasoning: str = ""


@dataclass
class AgenticRAGResult:
    """Final result from Agentic RAG loop.

    Attributes:
        response: Final generated response
        confidence: Confidence in the response
        iterations: Number of iterations taken
        states_visited: Sequence of states
        sub_queries: Decomposed sub-queries (if any)
        retrieval_sources: Sources used for retrieval
        documents_retrieved: Total documents retrieved
        generation_attempts: History of generation attempts
        evaluation_history: History of evaluations
        total_latency_ms: Total execution time
        metadata: Additional metadata
    """
    response: str
    confidence: float = 0.0
    iterations: int = 0
    states_visited: List[AgentState] = field(default_factory=list)
    sub_queries: List[SubQuery] = field(default_factory=list)
    retrieval_sources: List[str] = field(default_factory=list)
    documents_retrieved: int = 0
    generation_attempts: List[GenerationResult] = field(default_factory=list)
    evaluation_history: List[EvaluationResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class AgenticRAGPrompts:
    """Prompt templates for agentic RAG operations."""

    QUERY_ANALYSIS = """Analyze this query to determine:
1. Query type (factual, research, code, news, multi_hop, comparison, explanation, general)
2. Whether it needs decomposition into sub-queries
3. What information sources would be most useful

Query: {query}

Respond in this format:
QUERY_TYPE: [type]
NEEDS_DECOMPOSITION: [yes/no]
REASONING: [brief explanation]
SUGGESTED_SOURCES: [comma-separated list]"""

    QUERY_DECOMPOSITION = """Break down this complex query into simpler sub-queries that can be answered independently or in sequence.

Query: {query}

Previous context (if any): {context}

Instructions:
1. Identify the key components that need to be answered
2. Create specific, answerable sub-queries
3. Note any dependencies between sub-queries
4. Order by priority (answer first for best results)

Respond with sub-queries, one per line, prefixed with priority (1=highest):
Example:
1: What is X?
2: How does X relate to Y?
3: [depends on 1,2] What are the implications of X for Y?

Sub-queries:"""

    SELECT_TOOLS = """Given this query and available tools, select the best tools to use.

Query: {query}
Query Type: {query_type}

Available Tools:
{tools_description}

Instructions:
1. Consider what information sources would best answer this query
2. Select 1-3 tools that are most appropriate
3. Consider both relevance and coverage

Respond with tool names, comma-separated:
SELECTED_TOOLS: [tool1, tool2, ...]
REASONING: [why these tools]"""

    GENERATE_RESPONSE = """Generate a comprehensive response to the query using the provided context.

Query: {query}

Context:
{context}

Instructions:
1. Use information from the context to answer thoroughly
2. Cite specific parts of context when relevant
3. Acknowledge any gaps or limitations in the available information
4. Be accurate and well-structured

Response:"""

    EVALUATE_RESPONSE = """Evaluate this response for quality and completeness.

Query: {query}

Response:
{response}

Context used:
{context}

Evaluate on these criteria:
1. RELEVANCE: Does it directly address the query? (0.0-1.0)
2. COMPLETENESS: Does it cover all aspects? (0.0-1.0)
3. ACCURACY: Is it supported by the context? (0.0-1.0)
4. COHERENCE: Is it well-structured and clear? (0.0-1.0)

Respond with:
RELEVANCE: [score]
COMPLETENESS: [score]
ACCURACY: [score]
COHERENCE: [score]
ISSUES: [list any problems, or "none"]
SUGGESTIONS: [how to improve, or "none"]
DECISION: [COMPLETE/REFINE/RETRY_RETRIEVE/DECOMPOSE/FAIL]"""

    REFINE_RESPONSE = """Improve this response based on the identified issues.

Original Query: {query}

Original Response:
{response}

Issues:
{issues}

Suggestions:
{suggestions}

Additional Context:
{context}

Generate an improved response that addresses the identified issues:"""

    MERGE_SUBQUERY_RESPONSES = """Synthesize responses to sub-queries into a cohesive final answer.

Original Query: {query}

Sub-queries and Responses:
{subquery_responses}

Instructions:
1. Combine the information from all sub-query responses
2. Ensure logical flow and coherence
3. Eliminate redundancy while preserving important details
4. Address the original query comprehensively

Synthesized Response:"""


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class BaseRetrievalTool(ABC):
    """Base class for retrieval tools."""

    def __init__(self, name: str, description: str, query_types: List[str]):
        self._name = name
        self._description = description
        self._query_types = query_types

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def supported_query_types(self) -> List[str]:
        return self._query_types

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve documents."""
        pass


class ExaSearchTool(BaseRetrievalTool):
    """Wrapper for Exa search adapter."""

    def __init__(self, adapter: Any):
        super().__init__(
            name="exa",
            description="Neural AI search engine optimized for semantic understanding. Best for research, technical content, and finding authoritative sources.",
            query_types=["research", "factual", "code", "explanation"]
        )
        self.adapter = adapter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "auto",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using Exa."""
        try:
            result = await self.adapter.execute(
                "search",
                query=query,
                type=search_type,
                num_results=top_k,
                **kwargs
            )

            if not result.success:
                logger.warning(f"Exa search failed: {result.error}")
                return []

            documents = []
            for r in result.data.get("results", []):
                documents.append({
                    "content": r.get("text", r.get("highlight", r.get("title", ""))),
                    "score": r.get("score", 0.5),
                    "metadata": {
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "source": "exa",
                    }
                })

            return documents

        except Exception as e:
            logger.error(f"Exa retrieval error: {e}")
            return []


class TavilySearchTool(BaseRetrievalTool):
    """Wrapper for Tavily search adapter."""

    def __init__(self, adapter: Any):
        super().__init__(
            name="tavily",
            description="AI-optimized search engine with real-time web access. Best for current events, news, and general web content.",
            query_types=["news", "factual", "general", "research"]
        )
        self.adapter = adapter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        search_depth: str = "advanced",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using Tavily."""
        try:
            result = await self.adapter.execute(
                "search",
                query=query,
                max_results=top_k,
                search_depth=search_depth,
                include_answer=True,
                **kwargs
            )

            if not result.success:
                logger.warning(f"Tavily search failed: {result.error}")
                return []

            documents = []

            # Include the answer if available
            answer = result.data.get("answer")
            if answer:
                documents.append({
                    "content": answer,
                    "score": 0.95,
                    "metadata": {
                        "source": "tavily_answer",
                        "type": "synthesized"
                    }
                })

            for r in result.data.get("results", []):
                documents.append({
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.5),
                    "metadata": {
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "source": "tavily",
                    }
                })

            return documents

        except Exception as e:
            logger.error(f"Tavily retrieval error: {e}")
            return []


class MemorySearchTool(BaseRetrievalTool):
    """Wrapper for memory backend search."""

    def __init__(self, backend: Any, embedder: Optional[EmbeddingProvider] = None):
        super().__init__(
            name="memory",
            description="Internal knowledge base and memory search. Best for organizational knowledge, past conversations, and stored documents.",
            query_types=["factual", "explanation", "general"]
        )
        self.backend = backend
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search memory backend."""
        try:
            results = await self.backend.search(query, limit=top_k)

            documents = []
            for r in results:
                if hasattr(r, 'content'):
                    documents.append({
                        "content": r.content,
                        "score": getattr(r, 'score', 0.5),
                        "metadata": {
                            "id": getattr(r, 'id', ''),
                            "source": "memory",
                            **getattr(r, 'metadata', {})
                        }
                    })
                elif isinstance(r, dict):
                    documents.append({
                        "content": r.get("content", str(r)),
                        "score": r.get("score", 0.5),
                        "metadata": {
                            "source": "memory",
                            **r.get("metadata", {})
                        }
                    })

            return documents

        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []


class CodeSearchTool(BaseRetrievalTool):
    """Wrapper for code search tools (e.g., Context7)."""

    def __init__(self, adapter: Any):
        super().__init__(
            name="code_search",
            description="Search code repositories, documentation, and technical resources. Best for code examples, API documentation, and programming questions.",
            query_types=["code", "explanation", "factual"]
        )
        self.adapter = adapter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search code repositories."""
        try:
            # Try context7 or similar code search
            if hasattr(self.adapter, 'execute'):
                result = await self.adapter.execute(
                    "search",
                    query=query,
                    limit=top_k,
                    **kwargs
                )

                if not result.success:
                    return []

                documents = []
                for r in result.data.get("results", []):
                    documents.append({
                        "content": r.get("content", r.get("code", "")),
                        "score": r.get("score", 0.5),
                        "metadata": {
                            "source": "code_search",
                            "language": r.get("language", ""),
                            "file": r.get("file", ""),
                        }
                    })
                return documents

            return []

        except Exception as e:
            logger.error(f"Code search error: {e}")
            return []


class PerplexitySearchTool(BaseRetrievalTool):
    """Wrapper for Perplexity Sonar search adapter.

    Perplexity combines LLM capabilities with real-time web search,
    providing grounded responses with citations.

    Models:
        - sonar: Fast search and Q&A (128K context)
        - sonar-pro: Advanced multi-step queries, 2x citations (200K context)
        - sonar-reasoning-pro: Chain-of-thought reasoning
        - sonar-deep-research: Multi-step retrieval and synthesis
    """

    def __init__(self, adapter: Any):
        super().__init__(
            name="perplexity",
            description="AI-powered search with real-time web grounding and citations. Best for research questions, current events, and queries requiring synthesized answers from multiple sources.",
            query_types=["research", "news", "factual", "explanation", "multi_hop"]
        )
        self.adapter = adapter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "web",
        search_depth: str = "medium",
        use_pro: bool = False,
        use_reasoning: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using Perplexity Sonar.

        Args:
            query: Search query
            top_k: Maximum results (used for citation limit)
            search_mode: "web", "academic", or "sec"
            search_depth: "high", "medium", or "low"
            use_pro: Use sonar-pro for 2x citations
            use_reasoning: Use sonar-reasoning-pro for complex queries
            **kwargs: Additional arguments passed to adapter

        Returns:
            List of documents with content, score, and metadata
        """
        try:
            # Select operation based on flags
            if use_reasoning:
                operation = "reasoning"
            elif use_pro:
                operation = "pro"
            else:
                operation = "chat"

            result = await self.adapter.execute(
                operation,
                query=query,
                search_mode=search_mode,
                search_depth=search_depth,
                return_citations=True,
                **kwargs
            )

            if not result.success:
                logger.warning(f"Perplexity search failed: {result.error}")
                return []

            documents = []
            data = result.data or {}

            # Main synthesized content as primary document
            content = data.get("content", "")
            if content:
                documents.append({
                    "content": content,
                    "score": 0.95,  # High score for synthesized answer
                    "metadata": {
                        "source": "perplexity",
                        "type": "synthesized",
                        "model": data.get("model", "sonar"),
                    }
                })

            # Extract citations as additional documents
            citations = data.get("citations", [])
            for i, citation in enumerate(citations[:top_k]):
                if isinstance(citation, str):
                    # Citation is just a URL
                    documents.append({
                        "content": f"Source: {citation}",
                        "score": 0.8 - (i * 0.05),  # Decay score by position
                        "metadata": {
                            "source": "perplexity_citation",
                            "url": citation,
                            "position": i + 1,
                        }
                    })
                elif isinstance(citation, dict):
                    # Citation has more details
                    documents.append({
                        "content": citation.get("text", citation.get("title", str(citation))),
                        "score": 0.8 - (i * 0.05),
                        "metadata": {
                            "source": "perplexity_citation",
                            "url": citation.get("url", ""),
                            "title": citation.get("title", ""),
                            "position": i + 1,
                        }
                    })

            # Include related questions if available
            related = data.get("related_questions", [])
            if related:
                documents.append({
                    "content": "Related questions: " + "; ".join(related[:5]),
                    "score": 0.5,
                    "metadata": {
                        "source": "perplexity",
                        "type": "related_questions",
                    }
                })

            return documents

        except Exception as e:
            logger.error(f"Perplexity retrieval error: {e}")
            return []


class SerperSearchTool(BaseRetrievalTool):
    """Wrapper for Serper Google SERP API adapter.

    Serper provides fast, reliable Google search results with support for:
    - Web search (organic results, knowledge graph, people also ask)
    - Image search
    - News search
    - Video search
    - Places/Maps search
    - Google Scholar
    - Shopping results
    - Autocomplete suggestions
    """

    def __init__(self, adapter: Any):
        super().__init__(
            name="serper",
            description="Google SERP API providing web search results, knowledge graphs, news, images, and academic papers. Best for factual queries, current events, local information, and comprehensive web coverage.",
            query_types=["factual", "news", "research", "general", "comparison"]
        )
        self.adapter = adapter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "search",
        location: Optional[str] = None,
        time_range: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search using Serper Google SERP API.

        Args:
            query: Search query (supports Google operators: site:, filetype:, etc.)
            top_k: Maximum results to return
            search_type: "search", "news", "images", "videos", "scholar", "places"
            location: Location for local results (e.g., "San Francisco, CA")
            time_range: Time filter (qdr:h=hour, qdr:d=day, qdr:w=week, qdr:m=month, qdr:y=year)
            **kwargs: Additional arguments passed to adapter

        Returns:
            List of documents with content, score, and metadata
        """
        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "query": query,
                "num": top_k,
            }
            if location:
                request_kwargs["location"] = location
            if time_range:
                request_kwargs["tbs"] = time_range

            result = await self.adapter.execute(
                search_type,
                **request_kwargs,
                **kwargs
            )

            if not result.success:
                logger.warning(f"Serper search failed: {result.error}")
                return []

            documents = []
            data = result.data or {}

            # Handle different search types
            if search_type == "search":
                documents.extend(self._extract_web_results(data, top_k))
            elif search_type == "news":
                documents.extend(self._extract_news_results(data, top_k))
            elif search_type == "images":
                documents.extend(self._extract_image_results(data, top_k))
            elif search_type == "videos":
                documents.extend(self._extract_video_results(data, top_k))
            elif search_type == "scholar":
                documents.extend(self._extract_scholar_results(data, top_k))
            elif search_type == "places":
                documents.extend(self._extract_places_results(data, top_k))
            else:
                # Generic extraction for other types
                documents.extend(self._extract_web_results(data, top_k))

            return documents

        except Exception as e:
            logger.error(f"Serper retrieval error: {e}")
            return []

    def _extract_web_results(
        self,
        data: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract organic web search results."""
        documents = []

        # Knowledge graph (highest priority)
        kg = data.get("knowledgeGraph")
        if kg:
            kg_content = []
            if kg.get("title"):
                kg_content.append(f"**{kg['title']}**")
            if kg.get("type"):
                kg_content.append(f"Type: {kg['type']}")
            if kg.get("description"):
                kg_content.append(kg["description"])
            if kg.get("attributes"):
                for key, value in list(kg["attributes"].items())[:5]:
                    kg_content.append(f"- {key}: {value}")

            if kg_content:
                documents.append({
                    "content": "\n".join(kg_content),
                    "score": 0.98,
                    "metadata": {
                        "source": "serper_knowledge_graph",
                        "title": kg.get("title", ""),
                        "url": kg.get("website", kg.get("url", "")),
                    }
                })

        # Organic results
        organic = data.get("organic", [])
        for i, result in enumerate(organic[:top_k]):
            content_parts = []
            if result.get("title"):
                content_parts.append(f"**{result['title']}**")
            if result.get("snippet"):
                content_parts.append(result["snippet"])

            if content_parts:
                documents.append({
                    "content": "\n".join(content_parts),
                    "score": 0.9 - (i * 0.05),
                    "metadata": {
                        "source": "serper",
                        "url": result.get("link", ""),
                        "title": result.get("title", ""),
                        "position": result.get("position", i + 1),
                    }
                })

        # People Also Ask
        paa = data.get("peopleAlsoAsk", [])
        for i, qa in enumerate(paa[:3]):
            if qa.get("question") and qa.get("snippet"):
                documents.append({
                    "content": f"Q: {qa['question']}\nA: {qa['snippet']}",
                    "score": 0.7 - (i * 0.05),
                    "metadata": {
                        "source": "serper_paa",
                        "url": qa.get("link", ""),
                        "question": qa.get("question", ""),
                    }
                })

        # Related searches (lower priority)
        related = data.get("relatedSearches", [])
        if related:
            related_queries = [r.get("query", "") for r in related[:5] if r.get("query")]
            if related_queries:
                documents.append({
                    "content": "Related searches: " + ", ".join(related_queries),
                    "score": 0.4,
                    "metadata": {
                        "source": "serper_related",
                        "type": "related_searches",
                    }
                })

        return documents

    def _extract_news_results(
        self,
        data: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract news search results."""
        documents = []
        news = data.get("news", [])

        for i, article in enumerate(news[:top_k]):
            content_parts = []
            if article.get("title"):
                content_parts.append(f"**{article['title']}**")
            if article.get("source"):
                content_parts.append(f"Source: {article['source']}")
            if article.get("date"):
                content_parts.append(f"Date: {article['date']}")
            if article.get("snippet"):
                content_parts.append(article["snippet"])

            if content_parts:
                documents.append({
                    "content": "\n".join(content_parts),
                    "score": 0.9 - (i * 0.05),
                    "metadata": {
                        "source": "serper_news",
                        "url": article.get("link", ""),
                        "title": article.get("title", ""),
                        "date": article.get("date", ""),
                        "news_source": article.get("source", ""),
                    }
                })

        return documents

    def _extract_image_results(
        self,
        data: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract image search results."""
        documents = []
        images = data.get("images", [])

        for i, img in enumerate(images[:top_k]):
            content = f"Image: {img.get('title', 'Untitled')}"
            if img.get("source"):
                content += f" (Source: {img['source']})"

            documents.append({
                "content": content,
                "score": 0.85 - (i * 0.05),
                "metadata": {
                    "source": "serper_images",
                    "image_url": img.get("imageUrl", ""),
                    "thumbnail_url": img.get("thumbnailUrl", ""),
                    "page_url": img.get("link", ""),
                    "title": img.get("title", ""),
                    "width": img.get("imageWidth"),
                    "height": img.get("imageHeight"),
                }
            })

        return documents

    def _extract_video_results(
        self,
        data: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract video search results."""
        documents = []
        videos = data.get("videos", [])

        for i, video in enumerate(videos[:top_k]):
            content_parts = []
            if video.get("title"):
                content_parts.append(f"**{video['title']}**")
            if video.get("channel"):
                content_parts.append(f"Channel: {video['channel']}")
            if video.get("duration"):
                content_parts.append(f"Duration: {video['duration']}")
            if video.get("snippet"):
                content_parts.append(video["snippet"])

            if content_parts:
                documents.append({
                    "content": "\n".join(content_parts),
                    "score": 0.85 - (i * 0.05),
                    "metadata": {
                        "source": "serper_videos",
                        "url": video.get("link", ""),
                        "title": video.get("title", ""),
                        "channel": video.get("channel", ""),
                        "duration": video.get("duration", ""),
                        "date": video.get("date", ""),
                    }
                })

        return documents

    def _extract_scholar_results(
        self,
        data: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract Google Scholar results."""
        documents = []
        papers = data.get("organic", [])

        for i, paper in enumerate(papers[:top_k]):
            content_parts = []
            if paper.get("title"):
                content_parts.append(f"**{paper['title']}**")
            if paper.get("publicationInfo"):
                content_parts.append(f"Publication: {paper['publicationInfo']}")
            if paper.get("citedBy"):
                content_parts.append(f"Citations: {paper['citedBy']}")
            if paper.get("snippet"):
                content_parts.append(paper["snippet"])

            if content_parts:
                documents.append({
                    "content": "\n".join(content_parts),
                    "score": 0.9 - (i * 0.05),
                    "metadata": {
                        "source": "serper_scholar",
                        "url": paper.get("link", ""),
                        "pdf_url": paper.get("pdfUrl", ""),
                        "title": paper.get("title", ""),
                        "cited_by": paper.get("citedBy"),
                        "year": paper.get("year"),
                    }
                })

        return documents

    def _extract_places_results(
        self,
        data: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract local places results."""
        documents = []
        places = data.get("places", [])

        for i, place in enumerate(places[:top_k]):
            content_parts = []
            if place.get("title"):
                content_parts.append(f"**{place['title']}**")
            if place.get("address"):
                content_parts.append(f"Address: {place['address']}")
            if place.get("rating"):
                rating_str = f"Rating: {place['rating']}"
                if place.get("ratingCount"):
                    rating_str += f" ({place['ratingCount']} reviews)"
                content_parts.append(rating_str)
            if place.get("category"):
                content_parts.append(f"Category: {place['category']}")
            if place.get("phoneNumber"):
                content_parts.append(f"Phone: {place['phoneNumber']}")

            if content_parts:
                documents.append({
                    "content": "\n".join(content_parts),
                    "score": 0.9 - (i * 0.05),
                    "metadata": {
                        "source": "serper_places",
                        "url": place.get("website", ""),
                        "title": place.get("title", ""),
                        "address": place.get("address", ""),
                        "rating": place.get("rating"),
                        "latitude": place.get("latitude"),
                        "longitude": place.get("longitude"),
                        "cid": place.get("cid", ""),
                    }
                })

        return documents


# =============================================================================
# QUERY DECOMPOSER
# =============================================================================

class QueryDecomposer:
    """Decomposes complex queries into sub-queries."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = AgenticRAGPrompts()

    async def analyze_query(
        self,
        query: str
    ) -> Tuple[QueryType, bool, List[str]]:
        """Analyze query to determine type and if decomposition is needed.

        Returns:
            Tuple of (query_type, needs_decomposition, suggested_sources)
        """
        prompt = self.prompts.QUERY_ANALYSIS.format(query=query)

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.3
            )

            # Parse response
            query_type = QueryType.GENERAL
            needs_decomposition = False
            sources: List[str] = []

            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('QUERY_TYPE:'):
                    type_str = line.split(':', 1)[1].strip().lower()
                    try:
                        query_type = QueryType(type_str)
                    except ValueError:
                        query_type = QueryType.GENERAL

                elif line.startswith('NEEDS_DECOMPOSITION:'):
                    value = line.split(':', 1)[1].strip().lower()
                    needs_decomposition = value == 'yes'

                elif line.startswith('SUGGESTED_SOURCES:'):
                    sources_str = line.split(':', 1)[1].strip()
                    sources = [s.strip() for s in sources_str.split(',')]

            return query_type, needs_decomposition, sources

        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return QueryType.GENERAL, False, []

    async def decompose(
        self,
        query: str,
        context: str = ""
    ) -> List[SubQuery]:
        """Decompose query into sub-queries.

        Args:
            query: The complex query to decompose
            context: Any available context

        Returns:
            List of SubQuery objects
        """
        prompt = self.prompts.QUERY_DECOMPOSITION.format(
            query=query,
            context=context[:2000] if context else "None"
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=500,
                temperature=0.5
            )

            sub_queries: List[SubQuery] = []

            for line in response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Parse priority and query
                # Format: "1: What is X?" or "[depends on 1,2] 3: Question?"
                depends_on: List[str] = []

                # Check for dependencies
                if '[depends on' in line.lower():
                    import re
                    dep_match = re.search(r'\[depends on ([^\]]+)\]', line, re.I)
                    if dep_match:
                        deps = dep_match.group(1)
                        depends_on = [d.strip() for d in deps.split(',')]
                        line = re.sub(r'\[depends on [^\]]+\]', '', line).strip()

                # Parse priority and query
                if ':' in line:
                    parts = line.split(':', 1)
                    try:
                        priority = int(parts[0].strip())
                        query_text = parts[1].strip()
                    except ValueError:
                        priority = len(sub_queries) + 1
                        query_text = line

                    if query_text:
                        # Determine sub-query type
                        sub_type = self._infer_query_type(query_text)

                        sub_queries.append(SubQuery(
                            query=query_text,
                            query_type=sub_type,
                            priority=priority,
                            depends_on=depends_on
                        ))

            # Sort by priority
            sub_queries.sort(key=lambda q: q.priority)

            return sub_queries

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [SubQuery(query=query, query_type=QueryType.GENERAL, priority=1)]

    def _infer_query_type(self, query: str) -> QueryType:
        """Infer query type from text."""
        query_lower = query.lower()

        if any(w in query_lower for w in ['code', 'function', 'implement', 'api', 'syntax']):
            return QueryType.CODE
        elif any(w in query_lower for w in ['latest', 'recent', 'news', 'today', 'current']):
            return QueryType.NEWS
        elif any(w in query_lower for w in ['compare', 'difference', 'versus', 'vs', 'better']):
            return QueryType.COMPARISON
        elif any(w in query_lower for w in ['explain', 'why', 'how does', 'what is']):
            return QueryType.EXPLANATION
        elif any(w in query_lower for w in ['research', 'study', 'analysis', 'investigate']):
            return QueryType.RESEARCH
        else:
            return QueryType.FACTUAL


# =============================================================================
# TOOL SELECTOR
# =============================================================================

class ToolSelector:
    """Selects appropriate tools for a query."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[RetrievalTool]
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.prompts = AgenticRAGPrompts()

    def select_by_query_type(
        self,
        query_type: QueryType,
        max_tools: int = 3
    ) -> List[RetrievalTool]:
        """Select tools based on query type (rule-based)."""
        selected = []

        for tool in self.tools.values():
            if query_type.value in tool.supported_query_types:
                selected.append(tool)

        # Limit to max_tools
        return selected[:max_tools]

    async def select_with_llm(
        self,
        query: str,
        query_type: QueryType,
        max_tools: int = 3
    ) -> List[RetrievalTool]:
        """Select tools using LLM reasoning."""
        # Build tools description
        tools_desc = []
        for name, tool in self.tools.items():
            tools_desc.append(f"- {name}: {tool.description}")

        prompt = self.prompts.SELECT_TOOLS.format(
            query=query,
            query_type=query_type.value,
            tools_description="\n".join(tools_desc)
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=150,
                temperature=0.3
            )

            # Parse selected tools
            selected_names: List[str] = []
            for line in response.split('\n'):
                if line.strip().startswith('SELECTED_TOOLS:'):
                    tools_str = line.split(':', 1)[1].strip()
                    selected_names = [
                        t.strip().lower()
                        for t in tools_str.split(',')
                    ]
                    break

            # Map to actual tools
            selected = []
            for name in selected_names:
                if name in self.tools:
                    selected.append(self.tools[name])

            # Fallback to rule-based if LLM selection failed
            if not selected:
                selected = self.select_by_query_type(query_type, max_tools)

            return selected[:max_tools]

        except Exception as e:
            logger.warning(f"LLM tool selection failed: {e}")
            return self.select_by_query_type(query_type, max_tools)


# =============================================================================
# RESPONSE EVALUATOR
# =============================================================================

class ResponseEvaluator:
    """Evaluates generated responses for quality and completeness."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = AgenticRAGPrompts()

    async def evaluate(
        self,
        query: str,
        response: str,
        context: str,
        iteration: int,
        config: AgenticRAGConfig
    ) -> EvaluationResult:
        """Evaluate a generated response.

        Args:
            query: Original query
            response: Generated response
            context: Context used for generation
            iteration: Current iteration number
            config: Configuration

        Returns:
            EvaluationResult with scores and decision
        """
        prompt = self.prompts.EVALUATE_RESPONSE.format(
            query=query,
            response=response,
            context=context[:4000]  # Truncate for prompt
        )

        try:
            llm_response = await self.llm.generate(
                prompt,
                max_tokens=400,
                temperature=0.3
            )

            # Parse evaluation
            scores = {
                'relevance': 0.5,
                'completeness': 0.5,
                'accuracy': 0.5,
                'coherence': 0.5
            }
            issues: List[str] = []
            suggestions: List[str] = []
            decision = EvaluationDecision.REFINE

            for line in llm_response.split('\n'):
                line = line.strip()

                for key in scores:
                    if line.upper().startswith(key.upper() + ':'):
                        try:
                            value = float(line.split(':', 1)[1].strip())
                            scores[key] = max(0.0, min(1.0, value))
                        except (ValueError, IndexError):
                            pass

                if line.upper().startswith('ISSUES:'):
                    issues_str = line.split(':', 1)[1].strip()
                    if issues_str.lower() != 'none':
                        issues = [i.strip() for i in issues_str.split(',')]

                elif line.upper().startswith('SUGGESTIONS:'):
                    sugg_str = line.split(':', 1)[1].strip()
                    if sugg_str.lower() != 'none':
                        suggestions = [s.strip() for s in sugg_str.split(',')]

                elif line.upper().startswith('DECISION:'):
                    dec_str = line.split(':', 1)[1].strip().upper()
                    try:
                        decision = EvaluationDecision(dec_str.lower())
                    except ValueError:
                        # Infer decision from scores
                        pass

            # Calculate overall scores
            relevance_score = scores['relevance']
            completeness_score = scores['completeness']
            confidence = (
                scores['relevance'] * 0.3 +
                scores['completeness'] * 0.3 +
                scores['accuracy'] * 0.25 +
                scores['coherence'] * 0.15
            )

            # Override decision based on thresholds if needed
            if confidence >= config.confidence_threshold:
                decision = EvaluationDecision.COMPLETE
            elif iteration >= config.max_iterations:
                # Force complete at max iterations
                decision = EvaluationDecision.COMPLETE
            elif relevance_score < 0.3:
                decision = EvaluationDecision.RETRY_RETRIEVE
            elif completeness_score < config.completeness_threshold:
                decision = EvaluationDecision.REFINE

            return EvaluationResult(
                decision=decision,
                confidence=confidence,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                issues=issues,
                suggestions=suggestions
            )

        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,  # Default to complete on error
                confidence=0.5,
                relevance_score=0.5,
                completeness_score=0.5,
                issues=[f"Evaluation error: {e}"],
                suggestions=[]
            )


# =============================================================================
# RESULT FUSION
# =============================================================================

class ResultFusion:
    """Fuses results from multiple retrieval sources."""

    @staticmethod
    def rrf_fusion(
        result_lists: List[List[RetrievedContext]],
        k: int = 60
    ) -> List[RetrievedContext]:
        """Reciprocal Rank Fusion of multiple result lists.

        Args:
            result_lists: Lists of results from different sources
            k: RRF constant

        Returns:
            Fused list sorted by RRF score
        """
        if not result_lists:
            return []

        # Calculate RRF scores
        content_scores: Dict[str, Tuple[RetrievedContext, float]] = {}

        for results in result_lists:
            for rank, ctx in enumerate(results, start=1):
                # Use content hash as key for deduplication
                content_key = ctx.content[:200]
                rrf_score = 1.0 / (k + rank)

                if content_key in content_scores:
                    existing_ctx, existing_score = content_scores[content_key]
                    content_scores[content_key] = (existing_ctx, existing_score + rrf_score)
                else:
                    content_scores[content_key] = (ctx, rrf_score)

        # Sort by combined score
        sorted_results = sorted(
            content_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Update scores and return
        fused = []
        for ctx, score in sorted_results:
            ctx.score = score
            fused.append(ctx)

        return fused

    @staticmethod
    def weighted_fusion(
        result_lists: List[List[RetrievedContext]],
        weights: Optional[List[float]] = None
    ) -> List[RetrievedContext]:
        """Weighted fusion of results.

        Args:
            result_lists: Lists of results
            weights: Weights for each list

        Returns:
            Fused list
        """
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        content_scores: Dict[str, Tuple[RetrievedContext, float]] = {}

        for results, weight in zip(result_lists, weights):
            for ctx in results:
                content_key = ctx.content[:200]
                weighted_score = ctx.score * weight

                if content_key in content_scores:
                    existing_ctx, existing_score = content_scores[content_key]
                    content_scores[content_key] = (
                        existing_ctx,
                        max(existing_score, weighted_score)
                    )
                else:
                    content_scores[content_key] = (ctx, weighted_score)

        sorted_results = sorted(
            content_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        fused = []
        for ctx, score in sorted_results:
            ctx.score = score
            fused.append(ctx)

        return fused


# =============================================================================
# AGENTIC RAG MAIN CLASS
# =============================================================================

class AgenticRAG:
    """
    Agentic RAG Loop: Iterative Retrieval-Generation with Tool Augmentation.

    Implements a state-machine-based RAG loop that:
    - Decomposes complex queries into sub-queries
    - Dynamically selects appropriate retrieval tools
    - Iteratively retrieves, generates, and evaluates
    - Self-critiques and refines responses
    - Integrates with existing RAG strategies (Self-RAG, CRAG, HyDE, RAPTOR)

    State Machine:
        INIT -> PLAN -> RETRIEVE -> GENERATE -> EVALUATE -> (REFINE | COMPLETE)
                  ^                                  |
                  |<----- RETRY_RETRIEVE ------------|
                  |<----- DECOMPOSE -----------------|

    Example:
        >>> from core.rag.agentic_rag import AgenticRAG, AgenticRAGConfig
        >>>
        >>> config = AgenticRAGConfig(max_iterations=5, confidence_threshold=0.8)
        >>> agentic = AgenticRAG(
        ...     llm=my_llm,
        ...     tools=[exa_tool, tavily_tool, memory_tool],
        ...     config=config
        ... )
        >>>
        >>> result = await agentic.run("Complex question requiring multiple sources")
        >>> print(f"Response: {result.response}")
        >>> print(f"Confidence: {result.confidence}")
        >>> print(f"Iterations: {result.iterations}")
        >>> print(f"Sources used: {result.retrieval_sources}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[List[RetrievalTool]] = None,
        config: Optional[AgenticRAGConfig] = None,
        embedder: Optional[EmbeddingProvider] = None,
        rag_strategies: Optional[Dict[str, RAGStrategy]] = None,
    ):
        """Initialize Agentic RAG.

        Args:
            llm: LLM provider for generation and reasoning
            tools: List of retrieval tools
            config: Configuration options
            embedder: Optional embedding provider
            rag_strategies: Optional dict of RAG strategies (self_rag, crag, hyde, raptor)
        """
        self.llm = llm
        self.tools = tools or []
        self.config = config or AgenticRAGConfig()
        self.embedder = embedder
        self.rag_strategies = rag_strategies or {}

        # Initialize components
        self.decomposer = QueryDecomposer(llm)
        self.tool_selector = ToolSelector(llm, self.tools)
        self.evaluator = ResponseEvaluator(llm)
        self.prompts = AgenticRAGPrompts()

        # State tracking
        self._current_state: AgentState = AgentState.INIT
        self._cache: Dict[str, Any] = {}

    async def run(
        self,
        query: str,
        initial_context: Optional[List[str]] = None,
        **kwargs
    ) -> AgenticRAGResult:
        """Run the agentic RAG loop.

        Args:
            query: User query
            initial_context: Optional pre-provided context
            **kwargs: Additional arguments

        Returns:
            AgenticRAGResult with response and metadata
        """
        start_time = time.time()

        # Initialize result tracking
        states_visited: List[AgentState] = []
        sub_queries: List[SubQuery] = []
        generation_attempts: List[GenerationResult] = []
        evaluation_history: List[EvaluationResult] = []
        all_contexts: List[RetrievedContext] = []
        retrieval_sources: Set[str] = set()

        # Add initial context if provided
        if initial_context:
            for i, ctx in enumerate(initial_context):
                all_contexts.append(RetrievedContext(
                    content=ctx,
                    source="initial",
                    score=1.0,
                    metadata={"index": i}
                ))

        # State machine loop
        self._current_state = AgentState.INIT
        iteration = 0
        current_response = ""
        current_confidence = 0.0
        query_type = QueryType.GENERAL

        try:
            while iteration < self.config.max_iterations:
                states_visited.append(self._current_state)

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.timeout_seconds:
                    logger.warning(f"Agentic RAG timeout after {elapsed:.1f}s")
                    break

                # State transitions
                if self._current_state == AgentState.INIT:
                    self._current_state = AgentState.PLAN

                elif self._current_state == AgentState.PLAN:
                    # Analyze query and plan approach
                    query_type, needs_decomposition, suggested_sources = \
                        await self.decomposer.analyze_query(query)

                    if needs_decomposition and self.config.enable_query_decomposition:
                        context_text = self._build_context_text(all_contexts)
                        sub_queries = await self.decomposer.decompose(query, context_text)
                        logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")

                    self._current_state = AgentState.RETRIEVE

                elif self._current_state == AgentState.RETRIEVE:
                    # Select tools and retrieve
                    if self.config.enable_tool_selection and self.tools:
                        selected_tools = await self.tool_selector.select_with_llm(
                            query, query_type
                        )
                    else:
                        selected_tools = self.tools[:3]

                    # Retrieve for main query and sub-queries
                    queries_to_retrieve = [query]
                    if sub_queries:
                        queries_to_retrieve.extend([sq.query for sq in sub_queries[:3]])

                    retrieval_result = await self._retrieve_parallel(
                        queries_to_retrieve,
                        selected_tools
                    )

                    # Fuse results
                    if retrieval_result.contexts:
                        all_contexts.extend(retrieval_result.contexts)
                        retrieval_sources.update(retrieval_result.sources_used)

                    # Deduplicate and limit
                    all_contexts = self._deduplicate_contexts(
                        all_contexts,
                        self.config.max_total_documents
                    )

                    self._current_state = AgentState.GENERATE

                elif self._current_state == AgentState.GENERATE:
                    iteration += 1

                    # Build context
                    context_text = self._build_context_text(all_contexts)

                    # Generate response
                    if sub_queries and len(sub_queries) > 1:
                        # Handle sub-queries separately and merge
                        current_response = await self._generate_with_subqueries(
                            query, sub_queries, context_text
                        )
                    else:
                        current_response = await self._generate_response(
                            query, context_text
                        )

                    generation_attempts.append(GenerationResult(
                        response=current_response,
                        iteration=iteration,
                        contexts_used=len(all_contexts),
                        confidence=0.0,  # Will be updated after evaluation
                        reasoning=""
                    ))

                    self._current_state = AgentState.EVALUATE

                elif self._current_state == AgentState.EVALUATE:
                    # Evaluate response
                    context_text = self._build_context_text(all_contexts)
                    eval_result = await self.evaluator.evaluate(
                        query,
                        current_response,
                        context_text,
                        iteration,
                        self.config
                    )
                    evaluation_history.append(eval_result)

                    # Update confidence in latest generation
                    if generation_attempts:
                        generation_attempts[-1].confidence = eval_result.confidence

                    current_confidence = eval_result.confidence

                    # Decide next state
                    if eval_result.decision == EvaluationDecision.COMPLETE:
                        self._current_state = AgentState.COMPLETE
                    elif eval_result.decision == EvaluationDecision.RETRY_RETRIEVE:
                        self._current_state = AgentState.RETRIEVE
                    elif eval_result.decision == EvaluationDecision.DECOMPOSE:
                        self._current_state = AgentState.PLAN
                        # Force decomposition
                        sub_queries = []
                    elif eval_result.decision == EvaluationDecision.FAIL:
                        self._current_state = AgentState.ERROR
                    else:  # REFINE
                        self._current_state = AgentState.REFINE

                elif self._current_state == AgentState.REFINE:
                    # Refine response based on feedback
                    if evaluation_history:
                        last_eval = evaluation_history[-1]
                        current_response = await self._refine_response(
                            query,
                            current_response,
                            last_eval.issues,
                            last_eval.suggestions,
                            self._build_context_text(all_contexts)
                        )

                    # Return to evaluation
                    self._current_state = AgentState.EVALUATE

                elif self._current_state == AgentState.COMPLETE:
                    states_visited.append(AgentState.COMPLETE)
                    break

                elif self._current_state == AgentState.ERROR:
                    states_visited.append(AgentState.ERROR)
                    logger.error("Agentic RAG entered error state")
                    break

        except asyncio.TimeoutError:
            logger.error("Agentic RAG timed out")
            self._current_state = AgentState.ERROR

        except Exception as e:
            logger.error(f"Agentic RAG error: {e}")
            self._current_state = AgentState.ERROR

        # Build final result
        total_latency = (time.time() - start_time) * 1000

        return AgenticRAGResult(
            response=current_response,
            confidence=current_confidence,
            iterations=iteration,
            states_visited=states_visited,
            sub_queries=sub_queries,
            retrieval_sources=list(retrieval_sources),
            documents_retrieved=len(all_contexts),
            generation_attempts=generation_attempts,
            evaluation_history=evaluation_history,
            total_latency_ms=total_latency,
            metadata={
                "query_type": query_type.value,
                "final_state": self._current_state.value,
                "config": {
                    "max_iterations": self.config.max_iterations,
                    "confidence_threshold": self.config.confidence_threshold,
                }
            }
        )

    async def _retrieve_parallel(
        self,
        queries: List[str],
        tools: List[RetrievalTool]
    ) -> RetrievalResult:
        """Retrieve from multiple tools in parallel.

        Args:
            queries: List of queries to retrieve for
            tools: Tools to use

        Returns:
            Combined RetrievalResult
        """
        start_time = time.time()

        if not tools:
            return RetrievalResult(
                contexts=[],
                total_retrieved=0,
                sources_used=[],
                latency_ms=0
            )

        # Create retrieval tasks
        tasks = []
        task_metadata = []

        for query in queries:
            for tool in tools:
                task = tool.retrieve(
                    query,
                    top_k=self.config.top_k_per_source
                )
                tasks.append(task)
                task_metadata.append((query, tool.name))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        all_contexts: List[RetrievedContext] = []
        sources_used: Set[str] = set()

        for result, (query, tool_name) in zip(results, task_metadata):
            if isinstance(result, Exception):
                logger.warning(f"Retrieval failed for {tool_name}: {result}")
                continue

            if not result:
                continue

            sources_used.add(tool_name)

            for doc in result:
                all_contexts.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=tool_name,
                    score=doc.get("score", 0.5),
                    metadata=doc.get("metadata", {}),
                    tool_name=tool_name
                ))

        # Filter by relevance threshold
        all_contexts = [
            ctx for ctx in all_contexts
            if ctx.score >= self.config.relevance_threshold
        ]

        # Apply fusion if multiple sources
        if len(sources_used) > 1 and self.config.fusion_method == "rrf":
            # Group by source
            by_source: Dict[str, List[RetrievedContext]] = {}
            for ctx in all_contexts:
                if ctx.source not in by_source:
                    by_source[ctx.source] = []
                by_source[ctx.source].append(ctx)

            all_contexts = ResultFusion.rrf_fusion(
                list(by_source.values()),
                k=self.config.rrf_k
            )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            contexts=all_contexts,
            total_retrieved=len(all_contexts),
            sources_used=list(sources_used),
            latency_ms=latency_ms
        )

    def _deduplicate_contexts(
        self,
        contexts: List[RetrievedContext],
        max_count: int
    ) -> List[RetrievedContext]:
        """Deduplicate contexts and limit count."""
        seen: Set[str] = set()
        unique: List[RetrievedContext] = []

        # Sort by score first
        sorted_contexts = sorted(contexts, key=lambda c: c.score, reverse=True)

        for ctx in sorted_contexts:
            content_key = ctx.content[:200]
            if content_key not in seen:
                seen.add(content_key)
                unique.append(ctx)

            if len(unique) >= max_count:
                break

        return unique

    def _build_context_text(
        self,
        contexts: List[RetrievedContext]
    ) -> str:
        """Build context text from retrieved contexts."""
        if not contexts:
            return ""

        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            source_label = f"[Source {i}: {ctx.source}]"
            context_parts.append(f"{source_label}\n{ctx.content}")

        return "\n\n---\n\n".join(context_parts)

    async def _generate_response(
        self,
        query: str,
        context: str
    ) -> str:
        """Generate response using LLM."""
        # Check if a RAG strategy should be used
        if "self_rag" in self.rag_strategies:
            try:
                strategy = self.rag_strategies["self_rag"]
                result = await strategy.generate(
                    query,
                    context=[context] if context else None
                )
                if hasattr(result, 'response'):
                    return result.response
            except Exception as e:
                logger.warning(f"Self-RAG strategy failed: {e}")

        # Default generation
        prompt = self.prompts.GENERATE_RESPONSE.format(
            query=query,
            context=context if context else "No context available."
        )

        response = await self.llm.generate(
            prompt,
            max_tokens=self.config.generation_max_tokens,
            temperature=self.config.temperature
        )

        return response.strip()

    async def _generate_with_subqueries(
        self,
        main_query: str,
        sub_queries: List[SubQuery],
        context: str
    ) -> str:
        """Generate response by answering sub-queries and merging."""
        subquery_responses: List[Tuple[str, str]] = []

        # Answer each sub-query
        for sq in sub_queries:
            sq_response = await self._generate_response(sq.query, context)
            subquery_responses.append((sq.query, sq_response))

        # Merge responses
        subquery_text = "\n\n".join([
            f"Q: {q}\nA: {r}"
            for q, r in subquery_responses
        ])

        prompt = self.prompts.MERGE_SUBQUERY_RESPONSES.format(
            query=main_query,
            subquery_responses=subquery_text
        )

        merged = await self.llm.generate(
            prompt,
            max_tokens=self.config.generation_max_tokens,
            temperature=self.config.temperature
        )

        return merged.strip()

    async def _refine_response(
        self,
        query: str,
        response: str,
        issues: List[str],
        suggestions: List[str],
        context: str
    ) -> str:
        """Refine response based on evaluation feedback."""
        prompt = self.prompts.REFINE_RESPONSE.format(
            query=query,
            response=response,
            issues="\n".join(f"- {i}" for i in issues) if issues else "None",
            suggestions="\n".join(f"- {s}" for s in suggestions) if suggestions else "None",
            context=context[:4000]
        )

        refined = await self.llm.generate(
            prompt,
            max_tokens=self.config.generation_max_tokens,
            temperature=self.config.temperature * 0.8  # Lower temp for refinement
        )

        return refined.strip()

    def add_tool(self, tool: RetrievalTool) -> None:
        """Add a retrieval tool."""
        self.tools.append(tool)
        self.tool_selector = ToolSelector(self.llm, self.tools)

    def set_rag_strategy(self, name: str, strategy: RAGStrategy) -> None:
        """Set a RAG strategy."""
        self.rag_strategies[name] = strategy

    def clear_cache(self) -> None:
        """Clear internal cache."""
        self._cache.clear()


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

def create_agentic_rag(
    llm: LLMProvider,
    exa_adapter: Optional[Any] = None,
    tavily_adapter: Optional[Any] = None,
    memory_backend: Optional[Any] = None,
    code_search_adapter: Optional[Any] = None,
    perplexity_adapter: Optional[Any] = None,
    serper_adapter: Optional[Any] = None,
    config: Optional[AgenticRAGConfig] = None,
    self_rag: Optional[RAGStrategy] = None,
    crag: Optional[RAGStrategy] = None,
    hyde: Optional[RAGStrategy] = None,
    raptor: Optional[RAGStrategy] = None,
) -> AgenticRAG:
    """Factory function to create AgenticRAG with common tools.

    Args:
        llm: LLM provider
        exa_adapter: Optional Exa search adapter
        tavily_adapter: Optional Tavily search adapter
        memory_backend: Optional memory backend for search
        code_search_adapter: Optional code search adapter
        perplexity_adapter: Optional Perplexity Sonar adapter for AI-powered search
        serper_adapter: Optional Serper adapter for Google SERP results
        config: Configuration options
        self_rag: Optional Self-RAG strategy
        crag: Optional CRAG strategy
        hyde: Optional HyDE strategy
        raptor: Optional RAPTOR strategy

    Returns:
        Configured AgenticRAG instance

    Example:
        >>> from adapters.perplexity_adapter import PerplexityAdapter
        >>> from adapters.serper_adapter import SerperAdapter
        >>>
        >>> perplexity = PerplexityAdapter()
        >>> await perplexity.initialize({"api_key": "pplx-xxx"})
        >>>
        >>> serper = SerperAdapter()
        >>> await serper.initialize({"api_key": "serper-xxx"})
        >>>
        >>> agentic = create_agentic_rag(
        ...     llm=my_llm,
        ...     perplexity_adapter=perplexity,
        ...     serper_adapter=serper,
        ... )
    """
    tools: List[RetrievalTool] = []

    if exa_adapter is not None:
        tools.append(ExaSearchTool(exa_adapter))

    if tavily_adapter is not None:
        tools.append(TavilySearchTool(tavily_adapter))

    if memory_backend is not None:
        tools.append(MemorySearchTool(memory_backend))

    if code_search_adapter is not None:
        tools.append(CodeSearchTool(code_search_adapter))

    if perplexity_adapter is not None:
        tools.append(PerplexitySearchTool(perplexity_adapter))

    if serper_adapter is not None:
        tools.append(SerperSearchTool(serper_adapter))

    rag_strategies: Dict[str, RAGStrategy] = {}
    if self_rag:
        rag_strategies["self_rag"] = self_rag
    if crag:
        rag_strategies["crag"] = crag
    if hyde:
        rag_strategies["hyde"] = hyde
    if raptor:
        rag_strategies["raptor"] = raptor

    return AgenticRAG(
        llm=llm,
        tools=tools,
        config=config,
        rag_strategies=rag_strategies
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "AgenticRAG",
    # Configuration
    "AgenticRAGConfig",
    # Result types
    "AgenticRAGResult",
    "RetrievalResult",
    "EvaluationResult",
    "GenerationResult",
    "RetrievedContext",
    "SubQuery",
    # Enums
    "AgentState",
    "QueryType",
    "EvaluationDecision",
    # Tools
    "RetrievalTool",
    "BaseRetrievalTool",
    "ExaSearchTool",
    "TavilySearchTool",
    "MemorySearchTool",
    "CodeSearchTool",
    "PerplexitySearchTool",
    "SerperSearchTool",
    # Components
    "QueryDecomposer",
    "ToolSelector",
    "ResponseEvaluator",
    "ResultFusion",
    # Protocols
    "LLMProvider",
    "EmbeddingProvider",
    "RAGStrategy",
    # Prompts
    "AgenticRAGPrompts",
    # Factory
    "create_agentic_rag",
]
