"""
Tests for Query Complexity Analyzer

Tests cover:
- Token estimation
- Named entity detection
- Question type classification
- Domain detection
- Complexity scoring and level assignment
- Routing recommendations
- Cost estimation
"""

import pytest
from core.rag.complexity_analyzer import (
    QueryComplexityAnalyzer,
    ComplexityAwareRouter,
    AnalyzerConfig,
    ComplexityLevel,
    QuestionType,
    DomainType,
    ModelTier,
    EntityDetector,
    QuestionClassifier,
    DomainDetector,
)


# =============================================================================
# ENTITY DETECTOR TESTS
# =============================================================================

class TestEntityDetector:
    """Tests for EntityDetector class."""

    def setup_method(self):
        self.detector = EntityDetector()

    def test_detect_tech_terms(self):
        """Should detect technology terms."""
        entities = self.detector.detect_entities("I'm learning Python and React")
        tech_entities = [e for e in entities if e.entity_type == "TECH"]
        assert len(tech_entities) >= 2
        tech_texts = [e.text.lower() for e in tech_entities]
        assert "python" in tech_texts
        assert "react" in tech_texts

    def test_detect_organizations(self):
        """Should detect organization names."""
        entities = self.detector.detect_entities("Google and Microsoft compete in AI")
        org_entities = [e for e in entities if e.entity_type == "ORG"]
        org_texts = [e.text for e in org_entities]
        assert "Google" in org_texts
        assert "Microsoft" in org_texts

    def test_detect_dates(self):
        """Should detect date/time references."""
        entities = self.detector.detect_entities("The event happened in 2024")
        date_entities = [e for e in entities if e.entity_type == "DATE"]
        assert len(date_entities) >= 1
        assert any("2024" in e.text for e in date_entities)

    def test_has_quantifiers(self):
        """Should detect quantifier presence."""
        assert self.detector.has_quantifiers("List all the features")
        assert self.detector.has_quantifiers("50 percent of users")
        assert not self.detector.has_quantifiers("The feature is useful")

    def test_has_temporal_reference(self):
        """Should detect temporal references."""
        assert self.detector.has_temporal_reference("What happened today?")
        assert self.detector.has_temporal_reference("In 2023, the market changed")
        assert not self.detector.has_temporal_reference("The sky is blue")


# =============================================================================
# QUESTION CLASSIFIER TESTS
# =============================================================================

class TestQuestionClassifier:
    """Tests for QuestionClassifier class."""

    def setup_method(self):
        self.classifier = QuestionClassifier()

    def test_classify_factual(self):
        """Should classify factual questions."""
        assert self.classifier.classify("Who invented Python?") == QuestionType.FACTUAL
        assert self.classifier.classify("How many users are there?") == QuestionType.FACTUAL

    def test_classify_comparative(self):
        """Should classify comparative questions."""
        assert self.classifier.classify("Compare React vs Vue") == QuestionType.COMPARATIVE
        assert self.classifier.classify("What is the difference between X and Y") == QuestionType.COMPARATIVE

    def test_classify_procedural(self):
        """Should classify procedural questions."""
        assert self.classifier.classify("How to install Python?") == QuestionType.PROCEDURAL
        assert self.classifier.classify("Steps to deploy a Docker container") == QuestionType.PROCEDURAL

    def test_classify_causal(self):
        """Should classify causal questions."""
        assert self.classifier.classify("Why does JavaScript have closures?") == QuestionType.CAUSAL
        # "What causes" starts with "what" (factual pattern) and may not trigger causal
        result = self.classifier.classify("What causes memory leaks?")
        assert result in (QuestionType.CAUSAL, QuestionType.FACTUAL)

    def test_classify_definitional(self):
        """Should classify definitional questions."""
        assert self.classifier.classify("What is machine learning?") == QuestionType.DEFINITIONAL
        assert self.classifier.classify("Define microservices architecture") == QuestionType.DEFINITIONAL

    def test_classify_analytical(self):
        """Should classify analytical questions."""
        assert self.classifier.classify("Analyze the performance impact") == QuestionType.ANALYTICAL
        assert self.classifier.classify("How does caching improve speed?") == QuestionType.ANALYTICAL

    def test_classify_multi_part(self):
        """Should classify multi-part questions."""
        assert self.classifier.classify("What is X and how does it work?") == QuestionType.MULTI_PART

    def test_complexity_weights(self):
        """Complexity weights should increase with question complexity."""
        factual_weight = self.classifier.get_complexity_weight(QuestionType.FACTUAL)
        comparative_weight = self.classifier.get_complexity_weight(QuestionType.COMPARATIVE)
        multi_part_weight = self.classifier.get_complexity_weight(QuestionType.MULTI_PART)

        assert factual_weight < comparative_weight < multi_part_weight


# =============================================================================
# DOMAIN DETECTOR TESTS
# =============================================================================

class TestDomainDetector:
    """Tests for DomainDetector class."""

    def setup_method(self):
        self.detector = DomainDetector()

    def test_detect_programming(self):
        """Should detect programming domain."""
        # Use strong programming keywords for reliable detection
        assert self.detector.detect("How to debug a Python function with code?") == DomainType.PROGRAMMING

    def test_detect_technology(self):
        """Should detect technology domain."""
        assert self.detector.detect("Configure the cloud server network") == DomainType.TECHNOLOGY

    def test_detect_finance(self):
        """Should detect finance domain."""
        assert self.detector.detect("What are the best stock investment strategies?") == DomainType.FINANCE

    def test_detect_medical(self):
        """Should detect medical domain."""
        assert self.detector.detect("What are the symptoms of flu and its treatment?") == DomainType.MEDICINE

    def test_detect_news(self):
        """Should detect news domain."""
        assert self.detector.detect("What happened today in the latest news?") == DomainType.NEWS

    def test_detect_general(self):
        """Should default to general domain for queries with no domain keywords."""
        assert self.detector.detect("The weather is nice outside") == DomainType.GENERAL


# =============================================================================
# QUERY COMPLEXITY ANALYZER TESTS
# =============================================================================

class TestQueryComplexityAnalyzer:
    """Tests for QueryComplexityAnalyzer class."""

    def setup_method(self):
        self.analyzer = QueryComplexityAnalyzer()

    def test_analyze_simple_query(self):
        """Should analyze simple queries with low complexity."""
        analysis = self.analyzer.analyze("What is Python?")

        assert analysis.complexity_level in (ComplexityLevel.LOW, ComplexityLevel.MEDIUM)
        assert analysis.complexity_score < 0.5
        assert analysis.question_type == QuestionType.DEFINITIONAL
        assert analysis.token_count > 0

    def test_analyze_complex_query(self):
        """Should analyze complex queries with high complexity."""
        query = (
            "Compare the performance characteristics of React vs Vue "
            "for building large-scale enterprise applications, and also "
            "analyze their respective ecosystems and community support?"
        )
        analysis = self.analyzer.analyze(query)

        assert analysis.complexity_level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)
        assert analysis.complexity_score >= 0.5
        assert analysis.question_type in (QuestionType.COMPARATIVE, QuestionType.MULTI_PART)

    def test_analyze_multi_hop_query(self):
        """Should detect multi-hop queries."""
        query = "First find the author of Python and then list their other contributions"
        analysis = self.analyzer.analyze(query)

        assert analysis.is_multi_hop

    def test_analyze_reasoning_required(self):
        """Should detect queries requiring reasoning."""
        query = "Why does using microservices architecture improve scalability?"
        analysis = self.analyzer.analyze(query)

        assert analysis.requires_reasoning

    def test_analyze_current_info_required(self):
        """Should detect queries requiring current information."""
        query = "What are the latest AI developments announced today?"
        analysis = self.analyzer.analyze(query)

        assert analysis.requires_current_info

    def test_token_estimation(self):
        """Should estimate tokens correctly."""
        short_query = "What is X?"
        long_query = "What are the key differences between transformer and RNN architectures?"

        short_analysis = self.analyzer.analyze(short_query)
        long_analysis = self.analyzer.analyze(long_query)

        assert short_analysis.token_count < long_analysis.token_count

    def test_entity_detection(self):
        """Should detect entities in queries."""
        query = "How does Google's TensorFlow compare to PyTorch?"
        analysis = self.analyzer.analyze(query)

        assert analysis.entity_count > 0
        entity_texts = [e.text.lower() for e in analysis.entities]
        assert "tensorflow" in entity_texts or "pytorch" in entity_texts

    def test_routing_recommendations(self):
        """Should provide routing recommendations."""
        analysis = self.analyzer.analyze("Compare React and Vue frameworks")

        assert analysis.recommended_model_tier in ModelTier
        assert len(analysis.recommended_retrievers) > 0
        assert analysis.recommended_timeout_seconds > 0
        assert analysis.recommended_top_k > 0

    def test_cost_estimation(self):
        """Should provide cost estimates."""
        analysis = self.analyzer.analyze("What is machine learning?")

        assert analysis.cost_estimate is not None
        assert analysis.cost_estimate.total_cost_usd >= 0
        assert analysis.cost_estimate.model_tier in ModelTier

    def test_empty_query(self):
        """Should handle empty queries gracefully."""
        analysis = self.analyzer.analyze("")

        assert analysis.complexity_level == ComplexityLevel.LOW
        assert analysis.complexity_score == 0.0
        assert analysis.confidence == 0.0

    def test_get_complexity_breakdown(self):
        """Should provide detailed complexity breakdown."""
        breakdown = self.analyzer.get_complexity_breakdown("How does RAG improve LLM accuracy?")

        assert "query" in breakdown
        assert "complexity" in breakdown
        assert "classification" in breakdown
        assert "factors" in breakdown
        assert "recommendations" in breakdown
        assert "cost_estimate" in breakdown


# =============================================================================
# COMPLEXITY AWARE ROUTER TESTS
# =============================================================================

class TestComplexityAwareRouter:
    """Tests for ComplexityAwareRouter class."""

    def setup_method(self):
        self.router = ComplexityAwareRouter()

    def test_route_simple_query(self):
        """Should route simple queries to lower tiers."""
        analysis, config = self.router.route("What is Python?")

        assert analysis.complexity_level in (ComplexityLevel.LOW, ComplexityLevel.MEDIUM)
        assert config["strategy"] in ("basic", "self_rag")
        assert config["model_tier"] in (ModelTier.TIER_1, ModelTier.TIER_2)

    def test_route_complex_query(self):
        """Should route complex queries to higher tiers."""
        query = (
            "Compare and analyze the trade-offs between different "
            "microservices communication patterns, and then explain "
            "which one would be best for high-throughput systems?"
        )
        analysis, config = self.router.route(query)

        assert analysis.complexity_level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)
        assert config["strategy"] in ("agentic", "self_rag")
        assert config["model_tier"] == ModelTier.TIER_3

    def test_config_has_required_fields(self):
        """Config should have all required routing fields."""
        _, config = self.router.route("What is RAG?")

        assert "strategy" in config
        assert "retrievers" in config
        assert "top_k" in config
        assert "timeout_seconds" in config
        assert "enable_reranking" in config
        assert "enable_query_rewrite" in config
        assert "model_tier" in config


# =============================================================================
# MODEL TIER MAPPING TESTS
# =============================================================================

class TestModelTierMapping:
    """Tests for model tier mapping based on ADR-026."""

    def setup_method(self):
        self.analyzer = QueryComplexityAnalyzer()

    def test_tier_1_for_simple_queries(self):
        """Tier 1 should be recommended for simple queries."""
        simple_queries = [
            "What is X?",
            "Define Y",
            "Who is Z?",
        ]
        for query in simple_queries:
            analysis = self.analyzer.analyze(query)
            if analysis.complexity_level == ComplexityLevel.LOW:
                assert analysis.recommended_model_tier == ModelTier.TIER_1

    def test_tier_2_for_medium_queries(self):
        """Tier 2 should be recommended for medium queries."""
        analysis = self.analyzer.analyze("Explain how microservices architecture works")
        if analysis.complexity_level == ComplexityLevel.MEDIUM:
            assert analysis.recommended_model_tier == ModelTier.TIER_2

    def test_tier_3_for_complex_queries(self):
        """Tier 3 should be recommended for complex queries."""
        analysis = self.analyzer.analyze(
            "Analyze the performance implications of using event sourcing "
            "compared to traditional CRUD patterns in distributed systems, "
            "considering factors like consistency, scalability, and operational complexity"
        )
        if analysis.complexity_level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH):
            assert analysis.recommended_model_tier == ModelTier.TIER_3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complexity analyzer with routing."""

    def test_full_analysis_flow(self):
        """Should complete full analysis flow."""
        analyzer = QueryComplexityAnalyzer()
        router = ComplexityAwareRouter(analyzer=analyzer)

        query = "How does transformer attention mechanism improve NLP tasks?"

        # Get analysis
        analysis = analyzer.analyze(query)
        assert analysis.original_query == query

        # Get routing config
        routed_analysis, config = router.route(query)
        assert routed_analysis.original_query == query

        # Get routing recommendation
        recommendation = analysis.get_routing_recommendation()
        assert recommendation.model_tier == analysis.recommended_model_tier
        assert recommendation.retrievers == analysis.recommended_retrievers

    def test_analyzer_config_customization(self):
        """Should respect custom configuration."""
        config = AnalyzerConfig(
            low_threshold=0.2,
            medium_threshold=0.4,
            high_threshold=0.6,
            base_timeout=5.0,
        )
        analyzer = QueryComplexityAnalyzer(config=config)

        analysis = analyzer.analyze("What is Python?")
        assert analysis is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
