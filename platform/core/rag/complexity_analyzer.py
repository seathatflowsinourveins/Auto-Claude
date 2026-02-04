"""
Query Complexity Analyzer: Intelligent Routing for RAG Pipelines

Analyzes query characteristics to enable intelligent routing decisions:
- Model tier selection (Haiku vs Sonnet vs Opus)
- Retriever selection based on query type
- Timeout configuration scaling
- Cost estimation for budget management

Features:
- Token counting with multiple estimation methods
- Named entity recognition (rule-based, fast)
- Question type classification (factual, analytical, comparative, etc.)
- Domain detection for specialized routing
- Composite complexity scoring (LOW, MEDIUM, HIGH, VERY_HIGH)

Reference: ADR-026 3-Tier Model Routing

Integration:
    from core.rag.complexity_analyzer import (
        QueryComplexityAnalyzer,
        ComplexityLevel,
        QueryAnalysis,
    )

    analyzer = QueryComplexityAnalyzer()
    analysis = analyzer.analyze("Compare transformer vs RNN architectures")

    # Use for routing
    model_tier = analysis.recommended_model_tier
    timeout = analysis.recommended_timeout_seconds
    retrievers = analysis.recommended_retrievers
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ComplexityLevel(str, Enum):
    """Query complexity levels for routing decisions."""
    LOW = "low"           # Simple queries: Tier 1-2 (Agent Booster/Haiku)
    MEDIUM = "medium"     # Moderate queries: Tier 2 (Haiku)
    HIGH = "high"         # Complex queries: Tier 3 (Sonnet)
    VERY_HIGH = "very_high"  # Very complex: Tier 3 (Opus)


class QuestionType(str, Enum):
    """Classification of question types."""
    FACTUAL = "factual"           # Simple fact lookup ("What is X?")
    ANALYTICAL = "analytical"     # Requires analysis ("Why does X happen?")
    COMPARATIVE = "comparative"   # Compare/contrast ("X vs Y")
    PROCEDURAL = "procedural"     # How-to ("How to do X?")
    DEFINITIONAL = "definitional" # Definition ("Define X")
    CAUSAL = "causal"             # Cause/effect ("What causes X?")
    EVALUATIVE = "evaluative"     # Opinion/evaluation ("Is X good?")
    HYPOTHETICAL = "hypothetical" # What-if scenarios
    MULTI_PART = "multi_part"     # Multiple questions combined
    AGGREGATION = "aggregation"   # List/summarize ("What are all X?")
    TEMPORAL = "temporal"         # Time-based ("When did X happen?")
    SPATIAL = "spatial"           # Location-based ("Where is X?")
    UNKNOWN = "unknown"


class DomainType(str, Enum):
    """Domain classification for specialized routing."""
    GENERAL = "general"
    TECHNOLOGY = "technology"
    PROGRAMMING = "programming"
    SCIENCE = "science"
    MEDICINE = "medicine"
    LEGAL = "legal"
    FINANCE = "finance"
    ACADEMIC = "academic"
    NEWS = "news"
    CREATIVE = "creative"


class ModelTier(str, Enum):
    """Model tiers for routing (ADR-026)."""
    TIER_1 = "tier_1"  # Agent Booster (WASM) - <1ms, $0
    TIER_2 = "tier_2"  # Haiku - ~500ms, $0.0002
    TIER_3 = "tier_3"  # Sonnet/Opus - 2-5s, $0.003-$0.015


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EntityInfo:
    """Information about a detected entity."""
    text: str
    entity_type: str  # PERSON, ORG, TECH, LOCATION, DATE, etc.
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ComplexityFactors:
    """Individual factors contributing to complexity score."""
    token_count: int = 0
    entity_count: int = 0
    clause_count: int = 0
    question_count: int = 0
    negation_count: int = 0
    conditional_count: int = 0
    technical_term_count: int = 0
    has_comparison: bool = False
    has_temporal: bool = False
    has_quantifier: bool = False
    has_multi_hop_indicators: bool = False
    avg_word_length: float = 0.0
    unique_word_ratio: float = 0.0


@dataclass
class CostEstimate:
    """Estimated cost for processing a query."""
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model_tier: ModelTier
    notes: str = ""


@dataclass
class RoutingRecommendation:
    """Recommendations for query routing."""
    model_tier: ModelTier
    retrievers: List[str]
    timeout_seconds: float
    top_k: int
    enable_reranking: bool
    enable_query_expansion: bool
    enable_caching: bool


@dataclass
class QueryAnalysis:
    """Complete analysis of a query for routing decisions."""
    original_query: str
    complexity_level: ComplexityLevel
    complexity_score: float  # 0.0 - 1.0
    complexity_factors: ComplexityFactors

    # Classification
    question_type: QuestionType
    domain: DomainType

    # Token analysis
    token_count: int
    estimated_output_tokens: int

    # Entity analysis
    entities: List[EntityInfo] = field(default_factory=list)
    entity_count: int = 0

    # Routing recommendations
    recommended_model_tier: ModelTier = ModelTier.TIER_2
    recommended_retrievers: List[str] = field(default_factory=list)
    recommended_timeout_seconds: float = 30.0
    recommended_top_k: int = 5

    # Cost estimation
    cost_estimate: Optional[CostEstimate] = None

    # Additional metadata
    is_multi_hop: bool = False
    requires_reasoning: bool = False
    requires_current_info: bool = False
    confidence: float = 0.8

    def get_routing_recommendation(self) -> RoutingRecommendation:
        """Get complete routing recommendation."""
        return RoutingRecommendation(
            model_tier=self.recommended_model_tier,
            retrievers=self.recommended_retrievers,
            timeout_seconds=self.recommended_timeout_seconds,
            top_k=self.recommended_top_k,
            enable_reranking=self.complexity_level in (
                ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH
            ),
            enable_query_expansion=self.complexity_level != ComplexityLevel.LOW,
            enable_caching=not self.requires_current_info,
        )


# =============================================================================
# ANALYZER CONFIG
# =============================================================================

@dataclass
class AdaptiveTopKConfig:
    """Configuration for adaptive top-k retrieval based on query complexity.

    Adjusts retrieval depth to match query needs:
    - Simple queries need fewer documents to avoid noise
    - Complex queries benefit from more context

    Attributes:
        low_k: Documents for LOW complexity queries (default: 5)
        medium_k: Documents for MEDIUM complexity queries (default: 10)
        high_k: Documents for HIGH complexity queries (default: 15)
        very_high_k: Documents for VERY_HIGH complexity queries (default: 20)
        enabled: Whether adaptive top-k is enabled (default: True)
        min_k: Minimum allowed top-k (floor, default: 3)
        max_k: Maximum allowed top-k (ceiling, default: 50)
    """
    low_k: int = 5
    medium_k: int = 10
    high_k: int = 15
    very_high_k: int = 20
    enabled: bool = True
    min_k: int = 3
    max_k: int = 50

    def get_top_k(self, complexity: ComplexityLevel, override: Optional[int] = None) -> int:
        """Get top-k value for a complexity level with optional override.

        Args:
            complexity: The complexity level of the query
            override: Optional override value (bypasses adaptive logic)

        Returns:
            The top-k value to use, clamped to [min_k, max_k]
        """
        if override is not None:
            return max(self.min_k, min(self.max_k, override))

        if not self.enabled:
            return self.medium_k  # Default fallback when disabled

        mapping = {
            ComplexityLevel.LOW: self.low_k,
            ComplexityLevel.MEDIUM: self.medium_k,
            ComplexityLevel.HIGH: self.high_k,
            ComplexityLevel.VERY_HIGH: self.very_high_k,
        }
        k = mapping.get(complexity, self.medium_k)
        return max(self.min_k, min(self.max_k, k))


@dataclass
class AnalyzerConfig:
    """Configuration for the complexity analyzer."""
    # Thresholds for complexity levels
    low_threshold: float = 0.25
    medium_threshold: float = 0.50
    high_threshold: float = 0.75

    # Token estimation
    chars_per_token: float = 4.0
    output_token_multiplier: float = 2.5

    # Complexity weights
    token_weight: float = 0.15
    entity_weight: float = 0.15
    question_type_weight: float = 0.25
    structure_weight: float = 0.20
    domain_weight: float = 0.10
    linguistic_weight: float = 0.15

    # Timeout configuration (seconds)
    base_timeout: float = 10.0
    timeout_per_complexity: Dict[ComplexityLevel, float] = field(
        default_factory=lambda: {
            ComplexityLevel.LOW: 10.0,
            ComplexityLevel.MEDIUM: 20.0,
            ComplexityLevel.HIGH: 45.0,
            ComplexityLevel.VERY_HIGH: 90.0,
        }
    )

    # Cost per 1K tokens (USD) - ADR-026 reference
    model_costs: Dict[ModelTier, Tuple[float, float]] = field(
        default_factory=lambda: {
            ModelTier.TIER_1: (0.0, 0.0),           # Agent Booster (free)
            ModelTier.TIER_2: (0.00025, 0.00125),   # Haiku (input, output)
            ModelTier.TIER_3: (0.003, 0.015),       # Sonnet/Opus
        }
    )

    # Adaptive top-k configuration
    adaptive_top_k: AdaptiveTopKConfig = field(
        default_factory=AdaptiveTopKConfig
    )


# =============================================================================
# ENTITY DETECTOR
# =============================================================================

class EntityDetector:
    """Rule-based named entity detection for fast analysis."""

    # Technology/programming entities
    TECH_TERMS: Set[str] = {
        "python", "javascript", "typescript", "java", "rust", "go", "c++",
        "react", "vue", "angular", "node", "django", "flask", "fastapi",
        "kubernetes", "docker", "aws", "azure", "gcp", "terraform",
        "postgresql", "mongodb", "redis", "elasticsearch", "kafka",
        "llm", "gpt", "bert", "transformer", "rag", "langchain", "llamaindex",
        "api", "rest", "graphql", "grpc", "websocket", "microservices",
        "machine learning", "deep learning", "neural network", "nlp",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    }

    # Organization patterns
    ORG_PATTERNS: List[str] = [
        r"\b(?:Google|Microsoft|Amazon|Apple|Meta|OpenAI|Anthropic|"
        r"Facebook|Twitter|Netflix|Uber|Airbnb|Spotify|Salesforce|"
        r"IBM|Oracle|Intel|Nvidia|AMD|Tesla)\b",
    ]

    # Date/time patterns
    DATE_PATTERNS: List[str] = [
        r"\b\d{4}\b",  # Year
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b",
        r"\b(?:today|yesterday|tomorrow|recently|currently|now)\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    ]

    # Quantifier patterns
    QUANTIFIER_PATTERNS: List[str] = [
        r"\b(?:all|every|each|any|some|many|few|most|several)\b",
        r"\b\d+\s*(?:percent|%|million|billion|thousand)\b",
    ]

    def __init__(self):
        self._org_pattern = re.compile("|".join(self.ORG_PATTERNS), re.I)
        self._date_pattern = re.compile("|".join(self.DATE_PATTERNS), re.I)
        self._quantifier_pattern = re.compile(
            "|".join(self.QUANTIFIER_PATTERNS), re.I
        )

    def detect_entities(self, text: str) -> List[EntityInfo]:
        """Detect named entities in text."""
        entities: List[EntityInfo] = []
        text_lower = text.lower()

        # Detect technology terms
        for term in self.TECH_TERMS:
            pattern = rf"\b{re.escape(term)}\b"
            for match in re.finditer(pattern, text_lower):
                entities.append(EntityInfo(
                    text=text[match.start():match.end()],
                    entity_type="TECH",
                    start=match.start(),
                    end=match.end(),
                ))

        # Detect organizations
        for match in self._org_pattern.finditer(text):
            entities.append(EntityInfo(
                text=match.group(),
                entity_type="ORG",
                start=match.start(),
                end=match.end(),
            ))

        # Detect dates
        for match in self._date_pattern.finditer(text):
            entities.append(EntityInfo(
                text=match.group(),
                entity_type="DATE",
                start=match.start(),
                end=match.end(),
            ))

        # Detect proper nouns (capitalized words not at sentence start)
        proper_noun_pattern = r"(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]+"
        for match in re.finditer(proper_noun_pattern, text):
            word = match.group()
            if word.lower() not in self.TECH_TERMS and len(word) > 2:
                entities.append(EntityInfo(
                    text=word,
                    entity_type="PROPER_NOUN",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                ))

        # Deduplicate by position
        seen_positions: Set[Tuple[int, int]] = set()
        unique_entities: List[EntityInfo] = []
        for entity in entities:
            pos = (entity.start, entity.end)
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_entities.append(entity)

        return unique_entities

    def has_quantifiers(self, text: str) -> bool:
        """Check if text contains quantifiers."""
        return bool(self._quantifier_pattern.search(text))

    def has_temporal_reference(self, text: str) -> bool:
        """Check if text has temporal references."""
        return bool(self._date_pattern.search(text))


# =============================================================================
# QUESTION CLASSIFIER
# =============================================================================

class QuestionClassifier:
    """Classifies question types for routing decisions."""

    # Question type patterns (ordered by priority)
    PATTERNS: List[Tuple[str, QuestionType]] = [
        # Comparative
        (r"\bvs\.?\b|\bversus\b|\bcompare\b|\bcomparison\b|\bdifference\s+between\b|"
         r"\bbetter\s+than\b|\bworse\s+than\b|\bor\b.*\bwhich\b", QuestionType.COMPARATIVE),

        # Multi-part
        (r"\band\s+(?:also|what|how|why|when|where)\b|"
         r"\?\s*(?:and|also|what|how|why)\b|\?\s*\w+\s*\?", QuestionType.MULTI_PART),

        # Hypothetical
        (r"\bwhat\s+if\b|\bwould\s+happen\b|\bcould\s+happen\b|"
         r"\bsuppose\b|\bimagine\b|\bhypothetically\b", QuestionType.HYPOTHETICAL),

        # Causal
        (r"\bwhy\s+(?:does|do|is|are|did|was|were)\b|\bcause\b|\breason\b|"
         r"\bbecause\b|\bresult\s+in\b|\blead\s+to\b", QuestionType.CAUSAL),

        # Evaluative
        (r"\bshould\b|\brecommend\b|\bbest\b|\bworst\b|\bgood\b|\bbad\b|"
         r"\bpros?\s+and\s+cons?\b|\bworth\b|\badvice\b", QuestionType.EVALUATIVE),

        # Procedural
        (r"\bhow\s+(?:to|do|can|should)\b|\bsteps?\s+to\b|\bprocess\b|"
         r"\btutorial\b|\bguide\b|\binstruction\b", QuestionType.PROCEDURAL),

        # Aggregation
        (r"\blist\b|\ball\b|\bevery\b|\beach\b|\bexamples?\s+of\b|"
         r"\btypes?\s+of\b|\bkinds?\s+of\b", QuestionType.AGGREGATION),

        # Temporal
        (r"\bwhen\b|\bdate\b|\btimeline\b|\bhistory\b|\bfuture\b|"
         r"\brecent\b|\blast\b|\bfirst\b", QuestionType.TEMPORAL),

        # Spatial
        (r"\bwhere\b|\blocation\b|\bplace\b|\bregion\b|\bcountry\b|"
         r"\bcity\b|\baddress\b", QuestionType.SPATIAL),

        # Analytical
        (r"\banalyze\b|\banalysis\b|\bexplain\s+why\b|\binterpret\b|"
         r"\bevaluate\b|\bassess\b|\bhow\s+does\b", QuestionType.ANALYTICAL),

        # Definitional
        (r"\bwhat\s+is\b|\bdefine\b|\bdefinition\b|\bmeaning\b|"
         r"\bwhat\s+are\b|\bwhat\s+does\b.*\bmean\b", QuestionType.DEFINITIONAL),

        # Factual (default for simple questions)
        (r"\bwho\b|\bhow\s+much\b|\bhow\s+many\b|\bwhich\b|"
         r"\bwhat\b", QuestionType.FACTUAL),
    ]

    def __init__(self):
        self._compiled_patterns = [
            (re.compile(pattern, re.I), qtype)
            for pattern, qtype in self.PATTERNS
        ]

    def classify(self, query: str) -> QuestionType:
        """Classify the question type."""
        for pattern, qtype in self._compiled_patterns:
            if pattern.search(query):
                return qtype
        return QuestionType.UNKNOWN

    def get_complexity_weight(self, qtype: QuestionType) -> float:
        """Get complexity weight for question type."""
        weights = {
            QuestionType.FACTUAL: 0.1,
            QuestionType.DEFINITIONAL: 0.15,
            QuestionType.TEMPORAL: 0.2,
            QuestionType.SPATIAL: 0.2,
            QuestionType.PROCEDURAL: 0.4,
            QuestionType.AGGREGATION: 0.5,
            QuestionType.ANALYTICAL: 0.6,
            QuestionType.CAUSAL: 0.65,
            QuestionType.EVALUATIVE: 0.7,
            QuestionType.COMPARATIVE: 0.75,
            QuestionType.HYPOTHETICAL: 0.8,
            QuestionType.MULTI_PART: 0.9,
            QuestionType.UNKNOWN: 0.5,
        }
        return weights.get(qtype, 0.5)


# =============================================================================
# DOMAIN DETECTOR
# =============================================================================

class DomainDetector:
    """Detects the domain/topic area of a query."""

    DOMAIN_KEYWORDS: Dict[DomainType, Set[str]] = {
        DomainType.PROGRAMMING: {
            "code", "function", "class", "variable", "algorithm", "debug",
            "syntax", "compile", "runtime", "error", "exception", "api",
            "library", "framework", "git", "deploy", "test", "unit test",
        },
        DomainType.TECHNOLOGY: {
            "software", "hardware", "computer", "server", "cloud", "network",
            "database", "security", "encryption", "protocol", "system",
        },
        DomainType.SCIENCE: {
            "experiment", "hypothesis", "research", "study", "physics",
            "chemistry", "biology", "mathematics", "theorem", "proof",
        },
        DomainType.MEDICINE: {
            "medical", "health", "disease", "treatment", "diagnosis",
            "symptom", "patient", "doctor", "hospital", "medicine", "drug",
        },
        DomainType.LEGAL: {
            "law", "legal", "court", "judge", "attorney", "lawsuit",
            "contract", "regulation", "compliance", "statute", "rights",
        },
        DomainType.FINANCE: {
            "stock", "investment", "trading", "market", "financial",
            "bank", "loan", "interest", "portfolio", "dividend", "crypto",
        },
        DomainType.ACADEMIC: {
            "paper", "journal", "citation", "thesis", "dissertation",
            "professor", "university", "academic", "research", "peer review",
        },
        DomainType.NEWS: {
            "breaking", "latest", "today", "yesterday", "current events",
            "announcement", "reported", "according to", "sources say",
        },
        DomainType.CREATIVE: {
            "write", "story", "poem", "creative", "fiction", "character",
            "narrative", "plot", "artistic", "design", "creative writing",
        },
    }

    def __init__(self):
        # Build reverse lookup for efficiency
        self._keyword_to_domain: Dict[str, DomainType] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                self._keyword_to_domain[kw.lower()] = domain

    def detect(self, query: str) -> DomainType:
        """Detect the domain of the query."""
        query_lower = query.lower()
        domain_scores: Dict[DomainType, int] = {}

        # Count keyword matches per domain
        words = set(re.findall(r'\b\w+\b', query_lower))
        for word in words:
            if word in self._keyword_to_domain:
                domain = self._keyword_to_domain[word]
                domain_scores[domain] = domain_scores.get(domain, 0) + 1

        # Check for multi-word phrases
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if " " in kw and kw in query_lower:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 2

        if not domain_scores:
            return DomainType.GENERAL

        return max(domain_scores, key=domain_scores.get)

    def get_complexity_weight(self, domain: DomainType) -> float:
        """Get complexity weight for domain."""
        weights = {
            DomainType.GENERAL: 0.3,
            DomainType.NEWS: 0.35,
            DomainType.CREATIVE: 0.4,
            DomainType.TECHNOLOGY: 0.5,
            DomainType.PROGRAMMING: 0.55,
            DomainType.SCIENCE: 0.6,
            DomainType.FINANCE: 0.65,
            DomainType.LEGAL: 0.7,
            DomainType.MEDICINE: 0.75,
            DomainType.ACADEMIC: 0.7,
        }
        return weights.get(domain, 0.5)


# =============================================================================
# QUERY COMPLEXITY ANALYZER
# =============================================================================

class QueryComplexityAnalyzer:
    """
    Analyzes query complexity for intelligent routing decisions.

    Combines multiple analysis dimensions:
    - Token/length analysis
    - Named entity detection
    - Question type classification
    - Domain detection
    - Linguistic complexity (clauses, negations, conditionals)

    Usage:
        analyzer = QueryComplexityAnalyzer()
        analysis = analyzer.analyze("Compare React and Vue for building SPAs")

        print(f"Complexity: {analysis.complexity_level}")
        print(f"Model tier: {analysis.recommended_model_tier}")
        print(f"Timeout: {analysis.recommended_timeout_seconds}s")
    """

    # Multi-hop indicator patterns
    MULTI_HOP_PATTERNS: List[str] = [
        r"\band\s+then\b",
        r"\bafter\s+that\b",
        r"\bbased\s+on\s+(?:that|this|the)\b",
        r"\busing\s+(?:that|this|the)\b",
        r"\bfirst\b.*\bthen\b",
        r"\brelationship\s+between\b",
        r"\bhow\s+(?:does|do)\b.*\baffect\b",
    ]

    # Negation patterns
    NEGATION_PATTERNS: List[str] = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnone\b",
        r"\bwithout\b", r"\bn't\b", r"\bcan't\b", r"\bwon't\b",
        r"\bdon't\b", r"\bdoesn't\b", r"\bisn't\b", r"\baren't\b",
    ]

    # Conditional patterns
    CONDITIONAL_PATTERNS: List[str] = [
        r"\bif\b", r"\bwhen\b", r"\bwhile\b", r"\bunless\b",
        r"\bprovided\s+that\b", r"\bassuming\b", r"\bgiven\s+that\b",
    ]

    # Clause separators
    CLAUSE_SEPARATORS = r"[,;:()]|\band\b|\bor\b|\bbut\b|\bhowever\b|\bwhile\b"

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self.entity_detector = EntityDetector()
        self.question_classifier = QuestionClassifier()
        self.domain_detector = DomainDetector()

        # Compile patterns
        self._multi_hop_pattern = re.compile(
            "|".join(self.MULTI_HOP_PATTERNS), re.I
        )
        self._negation_pattern = re.compile(
            "|".join(self.NEGATION_PATTERNS), re.I
        )
        self._conditional_pattern = re.compile(
            "|".join(self.CONDITIONAL_PATTERNS), re.I
        )
        self._clause_pattern = re.compile(self.CLAUSE_SEPARATORS, re.I)

    def analyze(
        self,
        query: str,
        top_k_override: Optional[int] = None
    ) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.

        Args:
            query: The query string to analyze
            top_k_override: Optional override for top-k (bypasses adaptive logic)

        Returns:
            QueryAnalysis with complexity score and routing recommendations
        """
        if not query or not query.strip():
            return self._empty_analysis(query)

        query = query.strip()

        # Token analysis
        token_count = self._estimate_tokens(query)

        # Entity detection
        entities = self.entity_detector.detect_entities(query)
        entity_count = len(entities)

        # Question classification
        question_type = self.question_classifier.classify(query)

        # Domain detection
        domain = self.domain_detector.detect(query)

        # Linguistic complexity factors
        factors = self._analyze_linguistic_factors(query, token_count, entity_count)

        # Calculate composite complexity score
        complexity_score = self._calculate_complexity_score(
            factors, question_type, domain
        )

        # Determine complexity level
        complexity_level = self._score_to_level(complexity_score)

        # Check for special conditions
        is_multi_hop = bool(self._multi_hop_pattern.search(query))
        requires_reasoning = question_type in (
            QuestionType.ANALYTICAL, QuestionType.CAUSAL,
            QuestionType.HYPOTHETICAL, QuestionType.EVALUATIVE,
        )
        requires_current_info = (
            domain == DomainType.NEWS or
            self.entity_detector.has_temporal_reference(query)
        )

        # Get routing recommendations
        model_tier = self._recommend_model_tier(complexity_level, requires_reasoning)
        retrievers = self._recommend_retrievers(domain, question_type)
        timeout = self.config.timeout_per_complexity.get(
            complexity_level, self.config.base_timeout
        )
        top_k = self._recommend_top_k(complexity_level, top_k_override)

        # Estimate output tokens
        estimated_output = int(token_count * self.config.output_token_multiplier)
        if complexity_level == ComplexityLevel.VERY_HIGH:
            estimated_output = int(estimated_output * 1.5)

        # Cost estimation
        cost_estimate = self._estimate_cost(
            token_count, estimated_output, model_tier
        )

        return QueryAnalysis(
            original_query=query,
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            complexity_factors=factors,
            question_type=question_type,
            domain=domain,
            token_count=token_count,
            estimated_output_tokens=estimated_output,
            entities=entities,
            entity_count=entity_count,
            recommended_model_tier=model_tier,
            recommended_retrievers=retrievers,
            recommended_timeout_seconds=timeout,
            recommended_top_k=top_k,
            cost_estimate=cost_estimate,
            is_multi_hop=is_multi_hop,
            requires_reasoning=requires_reasoning,
            requires_current_info=requires_current_info,
            confidence=0.85,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based heuristic."""
        # Simple estimation: ~4 chars per token on average
        char_estimate = len(text) / self.config.chars_per_token

        # Word-based adjustment
        word_count = len(text.split())
        word_estimate = word_count * 1.3  # Average 1.3 tokens per word

        # Use weighted average
        return int((char_estimate + word_estimate) / 2)

    def _analyze_linguistic_factors(
        self, query: str, token_count: int, entity_count: int
    ) -> ComplexityFactors:
        """Analyze linguistic complexity factors."""
        query_lower = query.lower()
        words = query.split()
        unique_words = set(w.lower() for w in words)

        return ComplexityFactors(
            token_count=token_count,
            entity_count=entity_count,
            clause_count=len(self._clause_pattern.findall(query)) + 1,
            question_count=query.count("?"),
            negation_count=len(self._negation_pattern.findall(query_lower)),
            conditional_count=len(self._conditional_pattern.findall(query_lower)),
            technical_term_count=sum(
                1 for w in words
                if w.lower() in self.entity_detector.TECH_TERMS
            ),
            has_comparison="vs" in query_lower or "compare" in query_lower,
            has_temporal=self.entity_detector.has_temporal_reference(query),
            has_quantifier=self.entity_detector.has_quantifiers(query),
            has_multi_hop_indicators=bool(self._multi_hop_pattern.search(query)),
            avg_word_length=sum(len(w) for w in words) / len(words) if words else 0,
            unique_word_ratio=len(unique_words) / len(words) if words else 0,
        )

    def _calculate_complexity_score(
        self,
        factors: ComplexityFactors,
        question_type: QuestionType,
        domain: DomainType,
    ) -> float:
        """Calculate composite complexity score (0.0 - 1.0)."""
        # Token complexity (normalized to 0-1)
        token_score = min(factors.token_count / 100, 1.0)

        # Entity complexity
        entity_score = min(factors.entity_count / 10, 1.0)

        # Question type complexity
        qtype_score = self.question_classifier.get_complexity_weight(question_type)

        # Structural complexity
        structure_score = min(
            (factors.clause_count - 1) * 0.15 +
            factors.question_count * 0.2 +
            factors.negation_count * 0.1 +
            factors.conditional_count * 0.15 +
            (0.3 if factors.has_comparison else 0) +
            (0.2 if factors.has_multi_hop_indicators else 0),
            1.0
        )

        # Domain complexity
        domain_score = self.domain_detector.get_complexity_weight(domain)

        # Linguistic complexity
        linguistic_score = min(
            (factors.avg_word_length / 10) +
            (factors.technical_term_count * 0.1) +
            (0.2 if factors.unique_word_ratio > 0.8 else 0),
            1.0
        )

        # Weighted combination
        composite = (
            token_score * self.config.token_weight +
            entity_score * self.config.entity_weight +
            qtype_score * self.config.question_type_weight +
            structure_score * self.config.structure_weight +
            domain_score * self.config.domain_weight +
            linguistic_score * self.config.linguistic_weight
        )

        return min(composite, 1.0)

    def _score_to_level(self, score: float) -> ComplexityLevel:
        """Convert numeric score to complexity level."""
        if score < self.config.low_threshold:
            return ComplexityLevel.LOW
        elif score < self.config.medium_threshold:
            return ComplexityLevel.MEDIUM
        elif score < self.config.high_threshold:
            return ComplexityLevel.HIGH
        else:
            return ComplexityLevel.VERY_HIGH

    def _recommend_model_tier(
        self, complexity: ComplexityLevel, requires_reasoning: bool
    ) -> ModelTier:
        """Recommend model tier based on complexity."""
        # Upgrade tier if reasoning required
        if requires_reasoning and complexity == ComplexityLevel.LOW:
            complexity = ComplexityLevel.MEDIUM

        tier_mapping = {
            ComplexityLevel.LOW: ModelTier.TIER_1,
            ComplexityLevel.MEDIUM: ModelTier.TIER_2,
            ComplexityLevel.HIGH: ModelTier.TIER_3,
            ComplexityLevel.VERY_HIGH: ModelTier.TIER_3,
        }
        return tier_mapping.get(complexity, ModelTier.TIER_2)

    def _recommend_retrievers(
        self, domain: DomainType, question_type: QuestionType
    ) -> List[str]:
        """Recommend retrievers based on domain and question type."""
        retrievers = ["memory"]  # Always include memory

        # Domain-based recommendations
        if domain == DomainType.NEWS:
            retrievers.extend(["exa", "tavily"])
        elif domain in (DomainType.PROGRAMMING, DomainType.TECHNOLOGY):
            retrievers.extend(["exa", "context7"])
        elif domain == DomainType.ACADEMIC:
            retrievers.extend(["exa", "semantic_scholar"])
        else:
            retrievers.append("exa")

        # Question type adjustments
        if question_type == QuestionType.COMPARATIVE:
            if "tavily" not in retrievers:
                retrievers.append("tavily")
        elif question_type in (QuestionType.PROCEDURAL, QuestionType.AGGREGATION):
            if "context7" not in retrievers:
                retrievers.append("context7")

        return retrievers

    def _recommend_top_k(
        self,
        complexity: ComplexityLevel,
        override: Optional[int] = None
    ) -> int:
        """Recommend top_k based on complexity using adaptive configuration.

        Adaptive top-k retrieval adjusts retrieval depth to match query needs:
        - LOW complexity: k=5 (simple queries, avoid noise)
        - MEDIUM complexity: k=10 (standard queries)
        - HIGH complexity: k=15 (complex queries need more context)
        - VERY_HIGH complexity: k=20 (multi-hop/analytical queries)

        Args:
            complexity: The complexity level from analysis
            override: Optional override value (bypasses adaptive logic)

        Returns:
            The recommended top-k value
        """
        return self.config.adaptive_top_k.get_top_k(complexity, override)

    def _estimate_cost(
        self, input_tokens: int, output_tokens: int, model_tier: ModelTier
    ) -> CostEstimate:
        """Estimate cost for query processing."""
        input_rate, output_rate = self.config.model_costs.get(
            model_tier, (0.003, 0.015)
        )

        input_cost = (input_tokens / 1000) * input_rate
        output_cost = (output_tokens / 1000) * output_rate

        return CostEstimate(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            model_tier=model_tier,
        )

    def _empty_analysis(self, query: str) -> QueryAnalysis:
        """Return analysis for empty/invalid query."""
        return QueryAnalysis(
            original_query=query,
            complexity_level=ComplexityLevel.LOW,
            complexity_score=0.0,
            complexity_factors=ComplexityFactors(),
            question_type=QuestionType.UNKNOWN,
            domain=DomainType.GENERAL,
            token_count=0,
            estimated_output_tokens=0,
            entities=[],
            entity_count=0,
            recommended_model_tier=ModelTier.TIER_1,
            recommended_retrievers=["memory"],
            recommended_timeout_seconds=5.0,
            recommended_top_k=3,
            confidence=0.0,
        )

    def get_adaptive_top_k(
        self,
        query: str,
        override: Optional[int] = None
    ) -> int:
        """Get adaptive top-k value for a query.

        Convenience method that analyzes the query and returns the recommended
        top-k value based on complexity. Use this when you only need the top-k
        without the full analysis.

        Args:
            query: The query string
            override: Optional override (bypasses adaptive logic)

        Returns:
            The recommended top-k value

        Example:
            >>> analyzer = QueryComplexityAnalyzer()
            >>> k = analyzer.get_adaptive_top_k("What is Python?")  # Simple: k=5
            >>> k = analyzer.get_adaptive_top_k("Compare React vs Vue for SPAs")  # Complex: k=15
        """
        if override is not None:
            return self.config.adaptive_top_k.get_top_k(ComplexityLevel.MEDIUM, override)

        analysis = self.analyze(query)
        return analysis.recommended_top_k

    def get_complexity_breakdown(self, query: str) -> Dict[str, Any]:
        """Get detailed complexity breakdown for debugging/analysis."""
        analysis = self.analyze(query)

        return {
            "query": query,
            "complexity": {
                "level": analysis.complexity_level.value,
                "score": round(analysis.complexity_score, 3),
            },
            "classification": {
                "question_type": analysis.question_type.value,
                "domain": analysis.domain.value,
            },
            "factors": {
                "token_count": analysis.complexity_factors.token_count,
                "entity_count": analysis.complexity_factors.entity_count,
                "clause_count": analysis.complexity_factors.clause_count,
                "question_count": analysis.complexity_factors.question_count,
                "has_comparison": analysis.complexity_factors.has_comparison,
                "has_multi_hop": analysis.is_multi_hop,
                "requires_reasoning": analysis.requires_reasoning,
            },
            "recommendations": {
                "model_tier": analysis.recommended_model_tier.value,
                "retrievers": analysis.recommended_retrievers,
                "timeout_seconds": analysis.recommended_timeout_seconds,
                "top_k": analysis.recommended_top_k,
            },
            "cost_estimate": {
                "input_tokens": analysis.token_count,
                "output_tokens": analysis.estimated_output_tokens,
                "total_usd": round(
                    analysis.cost_estimate.total_cost_usd, 6
                ) if analysis.cost_estimate else 0,
            },
        }


# =============================================================================
# INTEGRATION WITH RAG PIPELINE
# =============================================================================

class ComplexityAwareRouter:
    """
    Router that uses complexity analysis for intelligent pipeline configuration.

    Integrates with RAGPipeline for dynamic strategy and retriever selection.
    """

    def __init__(
        self,
        analyzer: Optional[QueryComplexityAnalyzer] = None,
        tier_handlers: Optional[Dict[ModelTier, Any]] = None,
    ):
        self.analyzer = analyzer or QueryComplexityAnalyzer()
        self.tier_handlers = tier_handlers or {}

    def route(
        self,
        query: str,
        top_k_override: Optional[int] = None
    ) -> Tuple[QueryAnalysis, Dict[str, Any]]:
        """
        Route query based on complexity analysis.

        Args:
            query: The query string to route
            top_k_override: Optional override for top-k (bypasses adaptive logic)

        Returns:
            Tuple of (analysis, pipeline_config)
        """
        analysis = self.analyzer.analyze(query, top_k_override=top_k_override)

        # Build pipeline configuration with adaptive top-k
        config = {
            "strategy": self._select_strategy(analysis),
            "retrievers": analysis.recommended_retrievers,
            "top_k": analysis.recommended_top_k,
            "top_k_adaptive": self.analyzer.config.adaptive_top_k.enabled,
            "timeout_seconds": analysis.recommended_timeout_seconds,
            "enable_reranking": analysis.complexity_level in (
                ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH
            ),
            "enable_query_rewrite": analysis.complexity_level != ComplexityLevel.LOW,
            "enable_evaluation": analysis.complexity_level == ComplexityLevel.VERY_HIGH,
            "model_tier": analysis.recommended_model_tier,
            "complexity_level": analysis.complexity_level.value,
        }

        return analysis, config

    def _select_strategy(self, analysis: QueryAnalysis) -> str:
        """Select RAG strategy based on analysis."""
        if analysis.complexity_level == ComplexityLevel.LOW:
            return "basic"
        elif analysis.is_multi_hop or analysis.question_type == QuestionType.MULTI_PART:
            return "agentic"
        elif analysis.requires_reasoning:
            return "self_rag"
        elif analysis.requires_current_info:
            return "crag"
        elif analysis.complexity_level == ComplexityLevel.VERY_HIGH:
            return "agentic"
        else:
            return "basic"

    def get_handler(self, tier: ModelTier) -> Optional[Any]:
        """Get handler for a specific model tier."""
        return self.tier_handlers.get(tier)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main analyzer
    "QueryComplexityAnalyzer",
    "ComplexityAwareRouter",

    # Configuration
    "AnalyzerConfig",
    "AdaptiveTopKConfig",

    # Analysis results
    "QueryAnalysis",
    "ComplexityFactors",
    "CostEstimate",
    "RoutingRecommendation",
    "EntityInfo",

    # Enums
    "ComplexityLevel",
    "QuestionType",
    "DomainType",
    "ModelTier",

    # Component classes
    "EntityDetector",
    "QuestionClassifier",
    "DomainDetector",
]
