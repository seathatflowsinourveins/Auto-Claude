"""
Query Rewriter: Advanced Query Transformation for Improved RAG Retrieval

Features: Multi-query expansion, synonym/acronym expansion, step-back prompting,
query decomposition, intent classification, keyword extraction, semantic optimization.

Integration:
    from core.rag.query_rewriter import HybridQueryRewriter
    rewriter = HybridQueryRewriter(llm=my_llm)
    result = await rewriter.rewrite("Compare RAG vs fine-tuning")
"""

from __future__ import annotations
import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str: ...


class QueryRewriter(Protocol):
    """Protocol for query rewriters."""
    async def rewrite(self, query: str, **kwargs) -> "RewriteResult": ...


class QueryIntent(str, Enum):
    """Query intent classification."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    PROCEDURAL = "procedural"
    DEFINITION = "definition"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    LIST = "list"
    OPINION = "opinion"
    GENERAL = "general"


@dataclass
class QueryRewriterConfig:
    """Configuration for query rewriting."""
    num_expansions: int = 4
    enable_synonym_expansion: bool = True
    enable_step_back: bool = True
    enable_decomposition: bool = True
    max_sub_queries: int = 5
    extract_keywords: bool = True
    max_keywords: int = 10
    temperature: float = 0.7


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str
    intent: QueryIntent
    priority: int = 1
    is_dependent: bool = False


@dataclass
class RewriteResult:
    """Result of query rewriting."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    step_back_query: Optional[str] = None
    sub_queries: List[SubQuery] = field(default_factory=list)
    intent: QueryIntent = QueryIntent.GENERAL
    keywords: List[str] = field(default_factory=list)
    semantic_query: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    constraints: Dict[str, str] = field(default_factory=dict)
    is_complex: bool = False


class QueryRewriterPrompts:
    """Prompt templates for query rewriting."""
    MULTI_QUERY = "Generate {n} different query versions for search.\nOriginal: {query}\nRequirements: Vary phrasing, keep intent, use different terms.\nQueries:"
    STEP_BACK = "Generate a broader step-back question for: {query}\nStep-back question:"
    DECOMPOSE = "Break into sub-queries:\nQuery: {query}\nFormat: [PRIORITY] [INTENT] [DEPENDENT:yes/no] query\nSub-queries:"
    CLASSIFY_INTENT = "Classify intent (factual/comparison/explanation/procedural/definition/causal/temporal/list/opinion/general):\n{query}\nIntent:"
    EXTRACT_KEYWORDS = "Extract key search terms:\n{query}\nKeywords:"
    OPTIMIZE_SEMANTIC = "Optimize for vector search (expand acronyms, complete phrases):\n{query}\nOptimized:"


class RuleBasedQueryRewriter:
    """Fast rule-based query rewriter (no API calls)."""

    ACRONYMS = {
        "ml": "machine learning", "ai": "artificial intelligence", "nlp": "natural language processing",
        "llm": "large language model", "rag": "retrieval augmented generation", "api": "application programming interface",
        "db": "database", "sql": "structured query language", "js": "javascript", "ts": "typescript",
        "k8s": "kubernetes", "ci": "continuous integration", "cd": "continuous deployment",
        "tdd": "test driven development", "oop": "object oriented programming", "gpu": "graphics processing unit",
    }

    SYNONYMS = {
        "best": ["top", "optimal"], "fast": ["quick", "efficient"], "easy": ["simple"],
        "problem": ["issue", "error"], "fix": ["solve", "resolve"], "create": ["build", "develop"],
        "use": ["utilize", "implement"], "difference": ["comparison", "distinction"],
    }

    INTENT_PATTERNS = [
        (r"^what is\b|^define\b", QueryIntent.DEFINITION),
        (r"^how to\b|^steps to\b", QueryIntent.PROCEDURAL),
        (r"^why\b|^cause of\b", QueryIntent.CAUSAL),
        (r"\bvs\b|compare|difference between", QueryIntent.COMPARISON),
        (r"^when\b|timeline", QueryIntent.TEMPORAL),
        (r"^list\b|^what are the\b", QueryIntent.LIST),
        (r"^explain\b|^how does\b", QueryIntent.EXPLANATION),
        (r"\bbest\b|\brecommend\b", QueryIntent.OPINION),
        (r"^who\b|^where\b|^what\b", QueryIntent.FACTUAL),
    ]

    STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                  "do", "does", "did", "will", "would", "could", "should", "to", "of", "in", "for", "on",
                  "with", "at", "by", "from", "as", "into", "through", "what", "which", "who", "this", "that"}

    def __init__(self, config: Optional[QueryRewriterConfig] = None):
        self.config = config or QueryRewriterConfig()

    async def rewrite(self, query: str, **kwargs) -> RewriteResult:
        """Rewrite query using rules."""
        q_lower = query.lower().strip()
        intent = self._classify_intent(q_lower)
        is_complex = self._is_complex(q_lower)

        return RewriteResult(
            original_query=query,
            expanded_queries=self._expand_query(query)[:self.config.num_expansions],
            step_back_query=self._step_back(query, intent) if self.config.enable_step_back else None,
            sub_queries=self._decompose(query, intent)[:self.config.max_sub_queries] if is_complex and self.config.enable_decomposition else [],
            intent=intent,
            keywords=self._extract_keywords(query)[:self.config.max_keywords] if self.config.extract_keywords else [],
            semantic_query=self._optimize_semantic(query),
            entities=self._extract_entities(query),
            constraints=self._extract_constraints(query),
            is_complex=is_complex
        )

    def _classify_intent(self, query: str) -> QueryIntent:
        for pattern, intent in self.INTENT_PATTERNS:
            if re.search(pattern, query, re.I):
                return intent
        return QueryIntent.GENERAL

    def _is_complex(self, query: str) -> bool:
        return sum([" and " in query, " or " in query, query.count("?") > 1, len(query.split()) > 15]) >= 2

    def _expand_query(self, query: str) -> List[str]:
        expansions = [query]
        q_lower = query.lower()

        # Acronym expansion
        expanded = query
        for acr, full in self.ACRONYMS.items():
            if re.search(rf"\b{acr}\b", q_lower, re.I):
                expanded = re.sub(rf"\b{acr}\b", full, expanded, flags=re.I)
        if expanded != query:
            expansions.append(expanded)

        # Synonym expansion
        for word, syns in self.SYNONYMS.items():
            if word in q_lower:
                expansions.append(re.sub(rf"\b{word}\b", syns[0], query, flags=re.I))

        # Question reformulations
        if q_lower.startswith("what is "):
            topic = query[8:].rstrip("?")
            expansions.extend([f"{topic} definition", f"{topic} explained"])
        elif q_lower.startswith("how to "):
            topic = query[7:].rstrip("?")
            expansions.extend([f"{topic} tutorial", f"{topic} guide"])

        return list(dict.fromkeys(expansions))

    def _step_back(self, query: str, intent: QueryIntent) -> Optional[str]:
        topic = self._extract_topic(query)
        if not topic:
            return None
        templates = {
            QueryIntent.FACTUAL: f"What are the key facts about {topic}?",
            QueryIntent.COMPARISON: f"What are the main characteristics of {topic}?",
            QueryIntent.EXPLANATION: f"What are the fundamental concepts of {topic}?",
            QueryIntent.PROCEDURAL: f"What is the general approach to {topic}?",
        }
        return templates.get(intent, f"What should one know about {topic}?")

    def _extract_topic(self, query: str) -> Optional[str]:
        q = query.lower().rstrip("?")
        for s in ["what is", "how to", "why does", "how do i", "what are"]:
            if q.startswith(s):
                return q[len(s):].strip()
        words = [w for w in q.split() if w not in self.STOP_WORDS]
        return " ".join(words[:4]) if words else None

    def _extract_keywords(self, query: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', query.lower())
        keywords = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
        for w in list(keywords):
            if w in self.ACRONYMS:
                keywords.append(self.ACRONYMS[w])
        return list(dict.fromkeys(keywords))

    def _extract_entities(self, query: str) -> List[str]:
        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        entities += re.findall(r'"([^"]+)"', query)
        return list(dict.fromkeys(entities))

    def _extract_constraints(self, query: str) -> Dict[str, str]:
        constraints = {}
        if years := re.findall(r'\b(19|20)\d{2}\b', query):
            constraints["year"] = years[0]
        if versions := re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b', query):
            constraints["version"] = versions[0]
        for lang in ["python", "javascript", "java", "rust", "go"]:
            if lang in query.lower():
                constraints["language"] = lang
                break
        return constraints

    def _optimize_semantic(self, query: str) -> str:
        result = query
        for acr, full in self.ACRONYMS.items():
            result = re.sub(rf"\b{acr}\b", full, result, flags=re.I)
        return result.rstrip("?").strip()

    def _decompose(self, query: str, intent: QueryIntent) -> List[SubQuery]:
        parts = re.split(r'\s+(?:and|or|also)\s+', query, flags=re.I)
        return [SubQuery(query=p.strip().rstrip("?") + "?", intent=intent, priority=i+1, is_dependent=i > 0)
                for i, p in enumerate(parts) if len(p.strip()) > 10]


class LLMQueryRewriter:
    """LLM-powered query rewriter for intelligent transformations."""

    def __init__(self, llm: LLMProvider, config: Optional[QueryRewriterConfig] = None):
        self.llm = llm
        self.config = config or QueryRewriterConfig()
        self.prompts = QueryRewriterPrompts()
        self._rule_based = RuleBasedQueryRewriter(config)

    async def rewrite(self, query: str, **kwargs) -> RewriteResult:
        """Rewrite query using LLM."""
        tasks = [self._classify_intent(query), self._generate_expansions(query)]
        if self.config.enable_step_back:
            tasks.append(self._generate_step_back(query))
        if self.config.extract_keywords:
            tasks.append(self._extract_keywords(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        intent = results[0] if not isinstance(results[0], Exception) else QueryIntent.GENERAL
        expanded = results[1] if not isinstance(results[1], Exception) else [query]
        step_back = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
        keywords = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else []

        is_complex = self._is_complex(query, intent)
        sub_queries = await self._decompose(query) if is_complex and self.config.enable_decomposition else []
        semantic_query = await self._optimize_semantic(query)

        return RewriteResult(
            original_query=query,
            expanded_queries=[query] + expanded[:self.config.num_expansions - 1],
            step_back_query=step_back,
            sub_queries=sub_queries[:self.config.max_sub_queries],
            intent=intent,
            keywords=keywords[:self.config.max_keywords],
            semantic_query=semantic_query,
            entities=self._rule_based._extract_entities(query),
            constraints=self._rule_based._extract_constraints(query),
            is_complex=is_complex
        )

    async def _classify_intent(self, query: str) -> QueryIntent:
        try:
            resp = await self.llm.generate(self.prompts.CLASSIFY_INTENT.format(query=query), max_tokens=20, temperature=0.1)
            intent_str = resp.strip().lower()
            return QueryIntent(intent_str) if intent_str in [e.value for e in QueryIntent] else QueryIntent.GENERAL
        except:
            return self._rule_based._classify_intent(query.lower())

    async def _generate_expansions(self, query: str) -> List[str]:
        try:
            resp = await self.llm.generate(self.prompts.MULTI_QUERY.format(query=query, n=self.config.num_expansions),
                                           max_tokens=300, temperature=self.config.temperature)
            lines = [re.sub(r"^\d+[\.\)]\s*", "", l.strip()) for l in resp.strip().split("\n") if l.strip()]
            return [l for l in lines if l and l != query]
        except:
            return self._rule_based._expand_query(query)[1:]

    async def _generate_step_back(self, query: str) -> Optional[str]:
        try:
            return (await self.llm.generate(self.prompts.STEP_BACK.format(query=query), max_tokens=100, temperature=0.5)).strip()
        except:
            return None

    async def _extract_keywords(self, query: str) -> List[str]:
        try:
            resp = await self.llm.generate(self.prompts.EXTRACT_KEYWORDS.format(query=query), max_tokens=100, temperature=0.3)
            return [k.strip() for k in resp.split(",") if k.strip()]
        except:
            return self._rule_based._extract_keywords(query)

    async def _decompose(self, query: str) -> List[SubQuery]:
        try:
            resp = await self.llm.generate(self.prompts.DECOMPOSE.format(query=query), max_tokens=400, temperature=0.5)
            return self._parse_decomposition(resp)
        except:
            return self._rule_based._decompose(query, QueryIntent.GENERAL)

    def _parse_decomposition(self, response: str) -> List[SubQuery]:
        sub_queries = []
        for line in response.strip().split("\n"):
            if not line.strip():
                continue
            match = re.match(r"\[?(\d+)\]?\s*\[?(\w+)\]?\s*\[?(?:DEPENDENT:)?(yes|no)\]?\s*(.+)", line, re.I)
            if match:
                try:
                    intent = QueryIntent(match.group(2).lower())
                except ValueError:
                    intent = QueryIntent.GENERAL
                sub_queries.append(SubQuery(query=match.group(4).strip(), intent=intent,
                                           priority=int(match.group(1)), is_dependent=match.group(3).lower() == "yes"))
            elif len(line) > 10:
                sub_queries.append(SubQuery(query=line, intent=QueryIntent.GENERAL, priority=len(sub_queries) + 1))
        return sorted(sub_queries, key=lambda q: q.priority)

    async def _optimize_semantic(self, query: str) -> str:
        try:
            return (await self.llm.generate(self.prompts.OPTIMIZE_SEMANTIC.format(query=query), max_tokens=150, temperature=0.3)).strip()
        except:
            return self._rule_based._optimize_semantic(query)

    def _is_complex(self, query: str, intent: QueryIntent) -> bool:
        return sum([" and " in query.lower(), " or " in query.lower(), query.count("?") > 1,
                    len(query.split()) > 15, intent == QueryIntent.COMPARISON]) >= 2


class HybridQueryRewriter:
    """Hybrid rewriter combining LLM and rule-based approaches."""

    def __init__(self, llm: Optional[LLMProvider] = None, config: Optional[QueryRewriterConfig] = None, prefer_llm: bool = False):
        self.config = config or QueryRewriterConfig()
        self.prefer_llm = prefer_llm
        self._rule_based = RuleBasedQueryRewriter(self.config)
        self._llm_based = LLMQueryRewriter(llm, self.config) if llm else None

    async def rewrite(self, query: str, **kwargs) -> RewriteResult:
        """Rewrite using hybrid approach."""
        rule_result = await self._rule_based.rewrite(query)

        if not self._llm_based:
            return rule_result

        if not rule_result.is_complex and not self.prefer_llm and not self._needs_llm(query):
            return rule_result

        try:
            llm_result = await self._llm_based.rewrite(query, **kwargs)
            return self._merge(rule_result, llm_result)
        except Exception as e:
            logger.warning(f"LLM rewriting failed: {e}")
            return rule_result

    def _needs_llm(self, query: str) -> bool:
        return sum([len(query.split()) > 10, "?" in query,
                    any(w in query.lower() for w in ["explain", "compare", "why", "how"])]) >= 2

    def _merge(self, rule: RewriteResult, llm: RewriteResult) -> RewriteResult:
        expansions = list(llm.expanded_queries)
        for q in rule.expanded_queries:
            if q not in expansions:
                expansions.append(q)

        keywords = list(llm.keywords)
        for k in rule.keywords:
            if k not in keywords:
                keywords.append(k)

        return RewriteResult(
            original_query=llm.original_query,
            expanded_queries=expansions[:self.config.num_expansions + 2],
            step_back_query=llm.step_back_query or rule.step_back_query,
            sub_queries=llm.sub_queries or rule.sub_queries,
            intent=llm.intent,
            keywords=keywords[:self.config.max_keywords],
            semantic_query=llm.semantic_query or rule.semantic_query,
            entities=list(set(rule.entities + llm.entities)),
            constraints={**rule.constraints, **llm.constraints},
            is_complex=llm.is_complex or rule.is_complex
        )


__all__ = [
    "RuleBasedQueryRewriter", "LLMQueryRewriter", "HybridQueryRewriter",
    "QueryRewriterConfig", "RewriteResult", "SubQuery", "QueryIntent",
    "LLMProvider", "QueryRewriter", "QueryRewriterPrompts",
]
