"""
Tests for Adaptive Top-K Retrieval based on Query Complexity.

Verifies that retrieval depth dynamically adjusts based on query complexity:
- LOW complexity: k=5
- MEDIUM complexity: k=10
- HIGH complexity: k=15
- VERY_HIGH complexity: k=20
"""

import pytest
from core.rag.complexity_analyzer import (
    QueryComplexityAnalyzer,
    AnalyzerConfig,
    AdaptiveTopKConfig,
    ComplexityLevel,
)


class TestAdaptiveTopKConfig:
    """Tests for AdaptiveTopKConfig dataclass."""

    def test_default_values(self):
        """Test default adaptive top-k values."""
        config = AdaptiveTopKConfig()
        assert config.low_k == 5
        assert config.medium_k == 10
        assert config.high_k == 15
        assert config.very_high_k == 20
        assert config.enabled is True
        assert config.min_k == 3
        assert config.max_k == 50

    def test_get_top_k_by_complexity(self):
        """Test get_top_k returns correct values for each complexity level."""
        config = AdaptiveTopKConfig()

        assert config.get_top_k(ComplexityLevel.LOW) == 5
        assert config.get_top_k(ComplexityLevel.MEDIUM) == 10
        assert config.get_top_k(ComplexityLevel.HIGH) == 15
        assert config.get_top_k(ComplexityLevel.VERY_HIGH) == 20

    def test_get_top_k_with_override(self):
        """Test that override bypasses adaptive logic."""
        config = AdaptiveTopKConfig()

        # Override should return the override value regardless of complexity
        assert config.get_top_k(ComplexityLevel.LOW, override=25) == 25
        assert config.get_top_k(ComplexityLevel.VERY_HIGH, override=3) == 3

    def test_get_top_k_override_clamping(self):
        """Test that override values are clamped to [min_k, max_k]."""
        config = AdaptiveTopKConfig(min_k=3, max_k=50)

        # Override below min should be clamped
        assert config.get_top_k(ComplexityLevel.LOW, override=1) == 3

        # Override above max should be clamped
        assert config.get_top_k(ComplexityLevel.LOW, override=100) == 50

    def test_disabled_returns_medium(self):
        """Test that disabled config returns medium_k."""
        config = AdaptiveTopKConfig(enabled=False, medium_k=10)

        # Should always return medium_k when disabled
        assert config.get_top_k(ComplexityLevel.LOW) == 10
        assert config.get_top_k(ComplexityLevel.HIGH) == 10
        assert config.get_top_k(ComplexityLevel.VERY_HIGH) == 10

    def test_custom_values(self):
        """Test custom adaptive top-k configuration."""
        config = AdaptiveTopKConfig(
            low_k=3,
            medium_k=8,
            high_k=12,
            very_high_k=25,
        )

        assert config.get_top_k(ComplexityLevel.LOW) == 3
        assert config.get_top_k(ComplexityLevel.MEDIUM) == 8
        assert config.get_top_k(ComplexityLevel.HIGH) == 12
        assert config.get_top_k(ComplexityLevel.VERY_HIGH) == 25


class TestQueryComplexityAnalyzerAdaptiveTopK:
    """Tests for adaptive top-k integration with QueryComplexityAnalyzer."""

    def test_analyzer_with_adaptive_config(self):
        """Test analyzer uses adaptive top-k configuration."""
        adaptive_config = AdaptiveTopKConfig(
            low_k=5,
            medium_k=10,
            high_k=15,
            very_high_k=20,
        )
        analyzer_config = AnalyzerConfig(adaptive_top_k=adaptive_config)
        analyzer = QueryComplexityAnalyzer(config=analyzer_config)

        # Simple query should get low_k
        simple_analysis = analyzer.analyze("What is Python?")
        assert simple_analysis.recommended_top_k in [5, 10]  # LOW or MEDIUM

        # Complex query should get higher k
        complex_analysis = analyzer.analyze(
            "Compare the performance characteristics of React, Vue, and Angular "
            "for building large-scale enterprise applications with complex state "
            "management requirements, considering factors like bundle size, "
            "rendering performance, and developer productivity"
        )
        assert complex_analysis.recommended_top_k in [15, 20]  # HIGH or VERY_HIGH

    def test_get_adaptive_top_k_convenience_method(self):
        """Test the convenience method for getting adaptive top-k."""
        analyzer = QueryComplexityAnalyzer()

        # Simple query
        simple_k = analyzer.get_adaptive_top_k("What is Python?")
        assert 5 <= simple_k <= 10

        # Complex query
        complex_k = analyzer.get_adaptive_top_k(
            "Analyze the trade-offs between microservices and monolithic "
            "architecture patterns for a high-traffic e-commerce platform"
        )
        assert complex_k >= 10

    def test_get_adaptive_top_k_with_override(self):
        """Test convenience method respects override."""
        analyzer = QueryComplexityAnalyzer()

        # Override should be returned directly
        k = analyzer.get_adaptive_top_k("What is Python?", override=25)
        assert k == 25


class TestQueryComplexityLevels:
    """Test that different query types map to expected complexity levels."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default adaptive config."""
        return QueryComplexityAnalyzer()

    def test_simple_factual_queries(self, analyzer):
        """Simple factual queries should be LOW complexity."""
        queries = [
            "What is Python?",
            "When was JavaScript created?",
            "Who invented the internet?",
        ]
        for query in queries:
            analysis = analyzer.analyze(query)
            # Simple queries should be LOW or MEDIUM
            assert analysis.complexity_level in [
                ComplexityLevel.LOW,
                ComplexityLevel.MEDIUM,
            ], f"Query '{query}' was {analysis.complexity_level}"
            # Should get lower top-k
            assert analysis.recommended_top_k <= 10

    def test_comparison_queries(self, analyzer):
        """Comparison queries should be MEDIUM to HIGH complexity."""
        queries = [
            "Compare Python vs JavaScript for web development",
            "What are the differences between React and Vue?",
            "PostgreSQL versus MongoDB for microservices",
        ]
        for query in queries:
            analysis = analyzer.analyze(query)
            # Comparison queries should be at least MEDIUM
            assert analysis.complexity_level in [
                ComplexityLevel.MEDIUM,
                ComplexityLevel.HIGH,
                ComplexityLevel.VERY_HIGH,
            ], f"Query '{query}' was {analysis.complexity_level}"
            # Should get higher top-k
            assert analysis.recommended_top_k >= 10

    def test_multi_part_analytical_queries(self, analyzer):
        """Multi-part analytical queries should be HIGH to VERY_HIGH complexity."""
        queries = [
            "Explain how transformer architecture works and why it outperforms "
            "RNNs for sequence modeling, then discuss its implications for "
            "natural language understanding",
            "What are the security implications of using OAuth 2.0 with PKCE "
            "compared to the implicit flow, and how does this affect mobile "
            "application architecture decisions?",
        ]
        for query in queries:
            analysis = analyzer.analyze(query)
            # Complex analytical queries should be HIGH or VERY_HIGH
            assert analysis.complexity_level in [
                ComplexityLevel.HIGH,
                ComplexityLevel.VERY_HIGH,
            ], f"Query '{query}' was {analysis.complexity_level}"
            # Should get highest top-k values
            assert analysis.recommended_top_k >= 15


class TestAdaptiveTopKIntegration:
    """Integration tests for adaptive top-k with complexity routing."""

    def test_routing_recommendation_includes_adaptive_top_k(self):
        """Test that routing recommendation includes adaptive top-k."""
        from core.rag.complexity_analyzer import ComplexityAwareRouter

        router = ComplexityAwareRouter()

        # Test simple query
        analysis, config = router.route("What is Python?")
        assert "top_k" in config
        assert config["top_k_adaptive"] is True

        # Test complex query
        analysis, config = router.route(
            "Compare microservices vs monolith architecture patterns"
        )
        assert config["top_k"] >= 10

    def test_routing_with_override(self):
        """Test that routing respects top-k override."""
        from core.rag.complexity_analyzer import ComplexityAwareRouter

        router = ComplexityAwareRouter()

        # Override should be respected
        analysis, config = router.route("What is Python?", top_k_override=25)
        assert config["top_k"] == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
