"""
Tests for Finding Quality Module (GAP-02 Resolution)
====================================================

Tests garbage filtering, content extraction, and quality scoring.
"""

import pytest
from core.research.finding_quality import (
    FindingQualityProcessor,
    FindingQualityConfig,
    QualityScorer,
    ContentExtractor,
    TitleDetector,
    GarbagePatterns,
    process_exa_results,
    process_tavily_results,
    process_perplexity_results,
)


class TestGarbagePatterns:
    """Tests for garbage pattern detection."""

    def test_arxiv_patterns(self):
        """Test arXiv garbage detection."""
        garbage_texts = [
            "arXiv:2401.12345",
            "2401.12345",
            "[2401.12345]",
            "cs.AI",
            "stat.ML",
        ]
        patterns = GarbagePatterns.get_all_garbage_patterns()

        for text in garbage_texts:
            matches = any(p.match(text) for p in patterns)
            assert matches, f"Should detect garbage: {text}"

    def test_empty_patterns(self):
        """Test empty/placeholder detection."""
        garbage_texts = [
            "",
            "   ",
            "N/A",
            "null",
            "undefined",
            "none",
            "[]",
        ]
        patterns = GarbagePatterns.get_all_garbage_patterns()

        for text in garbage_texts:
            matches = any(p.match(text) for p in patterns)
            assert matches, f"Should detect garbage: '{text}'"

    def test_url_only_patterns(self):
        """Test URL-only detection."""
        garbage_texts = [
            "https://example.com/article",
            "www.example.com/page",
        ]
        patterns = GarbagePatterns.get_all_garbage_patterns()

        for text in garbage_texts:
            matches = any(p.match(text) for p in patterns)
            assert matches, f"Should detect garbage: {text}"

    def test_tag_only_patterns(self):
        """Test source tag only detection."""
        garbage_texts = [
            "[exa]",
            "[tavily]   ",
            "[perplexity]...",
        ]
        patterns = GarbagePatterns.get_all_garbage_patterns()

        for text in garbage_texts:
            matches = any(p.match(text) for p in patterns)
            assert matches, f"Should detect garbage: {text}"

    def test_valid_content_not_garbage(self):
        """Test that valid content is not flagged as garbage."""
        valid_texts = [
            "This is a valid finding about RAG architecture patterns.",
            "The study shows 45% improvement in retrieval accuracy.",
            "[exa] Vector databases enable fast similarity search.",
        ]
        patterns = GarbagePatterns.get_all_garbage_patterns()

        for text in valid_texts:
            matches = any(p.match(text) for p in patterns)
            assert not matches, f"Should not flag as garbage: {text}"


class TestTitleDetector:
    """Tests for title detection."""

    def test_title_only_detection(self):
        """Test detection of title-only content."""
        titles = [
            "Introduction to Vector Databases",
            "Best Practices for RAG",
            "A Comprehensive Guide",
        ]

        for title in titles:
            is_title, confidence = TitleDetector.is_likely_title(title)
            assert is_title, f"Should detect as title: {title}"
            assert confidence > 0.4, f"Should have reasonable confidence: {title}"

    def test_content_not_title(self):
        """Test that content is not flagged as title."""
        content_texts = [
            "The research demonstrates that hybrid retrieval combining dense and sparse methods achieves 23% better recall. This is particularly effective for domain-specific queries.",
            "We found that implementing semantic caching reduced API costs by 40% while maintaining 95% cache hit rates for similar queries.",
        ]

        for text in content_texts:
            is_title, confidence = TitleDetector.is_likely_title(text)
            assert not is_title, f"Should not detect as title: {text[:50]}..."

    def test_source_tag_stripping(self):
        """Test that source tags are stripped before analysis."""
        tagged = "[exa] Introduction to AI"
        is_title, _ = TitleDetector.is_likely_title(tagged)
        assert is_title, "Should strip tag and detect as title"


class TestQualityScorer:
    """Tests for quality scoring."""

    def test_garbage_gets_zero_score(self):
        """Test that garbage findings get zero score."""
        scorer = QualityScorer()

        garbage = [
            "arXiv:2401.12345",
            "",
            "[exa]",
            "N/A",
        ]

        for text in garbage:
            score = scorer.score(text)
            assert score.is_garbage, f"Should be garbage: {text}"
            assert score.total_score == 0.0, f"Garbage should have 0 score: {text}"

    def test_title_penalized(self):
        """Test that title-only findings are penalized."""
        scorer = QualityScorer()

        title = "[exa] Introduction to Vector Databases"
        score = scorer.score(title)

        assert score.is_title_only, "Should detect as title"
        assert score.total_score < 0.8, "Title should be penalized"

    def test_good_content_high_score(self):
        """Test that good content gets high score."""
        scorer = QualityScorer()

        good_content = (
            "[exa] The research demonstrates that hybrid retrieval combining dense "
            "and sparse methods achieves 23% better recall than either method alone. "
            "This improvement is consistent across different domain datasets."
        )

        score = scorer.score(good_content)
        assert not score.is_garbage, "Should not be garbage"
        assert not score.is_title_only, "Should not be title"
        assert score.total_score >= 0.7, "Good content should have high score"


class TestContentExtractor:
    """Tests for content extraction."""

    def test_extract_findings_from_text(self):
        """Test extraction of findings from source text."""
        extractor = ContentExtractor()

        text = """
        Vector databases have become essential for AI applications. Research shows that
        HNSW indexing provides the best balance of speed and accuracy for most use cases.
        In our benchmarks, we found that Qdrant achieved 99.5% recall at 1ms latency.

        The study demonstrates that hybrid search combining BM25 and dense vectors
        improves retrieval quality by 23%. This approach is particularly effective
        for technical documentation where exact keyword matches are important.
        """

        findings = extractor.extract_findings(text, source="test", max_findings=3)

        assert len(findings) > 0, "Should extract at least one finding"
        assert len(findings) <= 3, "Should respect max_findings"
        for f in findings:
            assert f.startswith("[test]"), "Should have source tag"
            assert len(f) >= 50, "Findings should have minimum length"

    def test_smart_truncate_word_boundary(self):
        """Test that truncation happens at word boundaries."""
        extractor = ContentExtractor()

        long_text = "This is a very long text that needs to be truncated at a proper word boundary not in the middle"

        # Set short max_length for test
        config = FindingQualityConfig(max_length=50, truncation_buffer=20)
        extractor.config = config

        truncated = extractor._smart_truncate(long_text)

        # Should not end with partial word
        assert not truncated.rstrip('.').endswith(' a'), "Should not cut mid-word"
        assert truncated.endswith('...'), "Should have ellipsis"

    def test_sentence_scoring(self):
        """Test sentence quality scoring."""
        extractor = ContentExtractor()

        # Insight sentence should score higher
        insight = "The research shows that implementing semantic caching reduced API costs by 40%."
        generic = "This is a sentence."

        insight_score = extractor._score_sentence(insight)
        generic_score = extractor._score_sentence(generic)

        assert insight_score > generic_score, "Insight should score higher"


class TestFindingQualityProcessor:
    """Tests for the main processor."""

    def test_process_findings_filters_garbage(self):
        """Test that garbage is filtered during processing."""
        processor = FindingQualityProcessor()

        raw_findings = [
            "[exa] arXiv:2401.12345",
            "[exa] This is a valid finding about RAG patterns that provides useful information.",
            "[tavily] N/A",
            "[tavily] The study shows 30% improvement in retrieval accuracy with hybrid methods.",
        ]

        result = processor.process_findings(raw_findings)

        assert len(result) == 2, f"Should keep only valid findings, got {len(result)}"
        assert all("arXiv" not in f for f in result), "Should filter arXiv garbage"
        assert all("N/A" not in f for f in result), "Should filter N/A"

    def test_process_findings_extracts_from_source(self):
        """Test extraction from source texts when raw findings are poor."""
        processor = FindingQualityProcessor()

        raw_findings = [
            "[exa] Article Title",  # Title only - poor quality
        ]

        source_texts = [
            "This research demonstrates that vector databases with HNSW indexing achieve "
            "99.5% recall at sub-millisecond latencies. The key finding is that proper "
            "parameter tuning can reduce memory usage by 40% without significant accuracy loss."
        ]

        result = processor.process_findings(
            raw_findings=raw_findings,
            source_texts=source_texts,
            source="exa"
        )

        # Should extract from source since title is poor quality
        assert len(result) > 0, "Should extract findings"
        # At least one finding should be from source extraction
        has_substantive = any(len(f) > 100 for f in result)
        assert has_substantive, "Should have substantive findings from source"

    def test_deduplication(self):
        """Test that duplicate findings are removed."""
        processor = FindingQualityProcessor()

        raw_findings = [
            "[exa] Vector databases enable fast similarity search for AI applications.",
            "[tavily] Vector databases enable fast similarity search for AI applications.",
        ]

        result = processor.process_findings(raw_findings)

        assert len(result) == 1, "Should deduplicate identical findings"

    def test_score_finding(self):
        """Test individual finding scoring."""
        processor = FindingQualityProcessor()

        good = (
            "[exa] The research demonstrates 45% improvement in retrieval "
            "accuracy when using hybrid search combining BM25 and dense vectors."
        )
        bad = "[exa]"

        good_score = processor.score_finding(good)
        bad_score = processor.score_finding(bad)

        assert good_score.total_score > bad_score.total_score
        assert bad_score.is_garbage


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_process_tavily_results(self):
        """Test Tavily result processing."""
        data = {
            "answer": (
                "RAG (Retrieval Augmented Generation) combines retrieval with "
                "generation to produce more accurate and grounded responses. "
                "Studies show this approach reduces hallucinations by up to 50%."
            ),
            "results": [
                {"title": "RAG Guide", "content": "Some content here."},
            ]
        }

        findings = process_tavily_results(data)

        assert len(findings) > 0, "Should produce findings"
        assert findings[0].startswith("[tavily]"), "Should have tavily tag"

    def test_process_perplexity_results(self):
        """Test Perplexity result processing."""
        content = (
            "Vector databases are specialized systems for storing and querying "
            "high-dimensional vectors. The most common indexing algorithm is HNSW, "
            "which provides excellent recall-latency tradeoffs. Research shows that "
            "proper parameter tuning can improve query performance by 3x."
        )

        findings = process_perplexity_results(content)

        assert len(findings) > 0, "Should extract findings"
        assert all("[perplexity]" in f for f in findings), "Should have perplexity tag"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self):
        """Test handling of empty input."""
        processor = FindingQualityProcessor()

        result = processor.process_findings([])
        assert result == [], "Should return empty list"

        result = processor.process_findings([""])
        assert result == [], "Should filter empty strings"

    def test_unicode_content(self):
        """Test handling of unicode content."""
        processor = FindingQualityProcessor()

        unicode_finding = (
            "[exa] The research shows that embedding models handle "
            "multilingual content effectively: \u4e2d\u6587, \u65e5\u672c\u8a9e, and \ud55c\uad6d\uc5b4."
        )

        result = processor.process_findings([unicode_finding])
        assert len(result) > 0, "Should handle unicode"

    def test_very_long_content(self):
        """Test handling of very long content."""
        processor = FindingQualityProcessor()

        long_text = "This is a finding. " * 100  # Very long

        result = processor.smart_truncate(long_text, max_length=200)

        assert len(result) <= 210, "Should truncate long content"
        assert result.endswith('...'), "Should have ellipsis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
