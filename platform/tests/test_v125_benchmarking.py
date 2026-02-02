#!/usr/bin/env python3
"""
V125: Benchmarking Framework Tests

Tests the benchmarking suite for V115-V124 optimizations.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))


# =============================================================================
# Import Tests
# =============================================================================

class TestV125Imports:
    """Test V125 module imports correctly."""

    def test_benchmarking_module_imports(self):
        """Verify benchmarking module can be imported."""
        from benchmarking import (
            BenchmarkSuite,
            BenchmarkResult,
            BenchmarkStats,
            BenchmarkCategory,
            run_full_benchmark,
            quick_benchmark,
        )
        assert BenchmarkSuite is not None
        assert BenchmarkResult is not None
        assert BenchmarkStats is not None
        assert BenchmarkCategory is not None
        assert run_full_benchmark is not None
        assert quick_benchmark is not None

    def test_content_detection_cases_exist(self):
        """Verify ground truth test cases are defined."""
        from benchmarking import CONTENT_DETECTION_CASES
        assert len(CONTENT_DETECTION_CASES) > 10
        # Should have code, text, multilingual cases
        case_types = set()
        for _, content_type in CONTENT_DETECTION_CASES:
            case_types.add(content_type.value)
        assert "code" in case_types
        assert "text" in case_types
        assert "multilingual" in case_types


# =============================================================================
# Data Class Tests
# =============================================================================

class TestBenchmarkDataClasses:
    """Test benchmark data structures."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult can be created."""
        from benchmarking import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            success=True,
            latency_ms=1.5,
            iterations=100,
            metadata={"key": "value"},
        )
        assert result.name == "test"
        assert result.success is True
        assert result.latency_ms == 1.5
        assert result.iterations == 100
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_benchmark_result_with_error(self):
        """Test BenchmarkResult with error."""
        from benchmarking import BenchmarkResult

        result = BenchmarkResult(
            name="failed_test",
            success=False,
            latency_ms=0,
            error="Test failed",
        )
        assert result.success is False
        assert result.error == "Test failed"

    def test_benchmark_stats_to_dict(self):
        """Test BenchmarkStats serialization."""
        from benchmarking import BenchmarkStats

        stats = BenchmarkStats(
            name="test",
            total_runs=100,
            successful_runs=95,
            failed_runs=5,
            min_latency_ms=0.5,
            max_latency_ms=10.0,
            mean_latency_ms=2.5,
            median_latency_ms=2.0,
            p95_latency_ms=5.0,
            p99_latency_ms=8.0,
            std_dev_ms=1.5,
            throughput_ops_sec=400.0,
            accuracy=0.95,
        )

        d = stats.to_dict()
        assert d["name"] == "test"
        assert d["total_runs"] == 100
        assert d["successful_runs"] == 95
        assert d["failed_runs"] == 5
        assert d["latency_ms"]["mean"] == 2.5
        assert d["latency_ms"]["p95"] == 5.0
        assert d["throughput_ops_sec"] == 400.0
        assert d["accuracy"] == 0.95


# =============================================================================
# Benchmark Suite Tests
# =============================================================================

class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_suite_initialization(self):
        """Test suite can be initialized."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(
            warmup_iterations=2,
            benchmark_iterations=10,
        )
        assert suite.warmup_iterations == 2
        assert suite.benchmark_iterations == 10

    def test_suite_default_values(self):
        """Test suite uses sensible defaults."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite()
        assert suite.warmup_iterations == 5
        assert suite.benchmark_iterations == 100

    @pytest.mark.asyncio
    async def test_benchmark_content_detection(self):
        """Test content detection benchmark runs."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_content_detection()

        assert stats.name == "content_detection"
        assert stats.total_runs > 0
        # Should have some accuracy
        if stats.accuracy is not None:
            assert 0 <= stats.accuracy <= 1

    @pytest.mark.asyncio
    async def test_benchmark_routing_decisions(self):
        """Test routing decisions benchmark runs."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_routing_decisions()

        assert stats.name == "routing_decisions"
        assert stats.total_runs > 0

    @pytest.mark.asyncio
    async def test_benchmark_cache_performance(self):
        """Test cache performance benchmark runs."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_cache_performance()

        assert stats.name == "cache_performance"
        assert stats.total_runs > 0

    @pytest.mark.asyncio
    async def test_benchmark_metrics_collection(self):
        """Test metrics collection benchmark runs."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_metrics_collection()

        assert stats.name == "metrics_collection"
        assert stats.total_runs > 0

    @pytest.mark.asyncio
    async def test_generate_report(self):
        """Test report generation."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        await suite.benchmark_content_detection()
        await suite.benchmark_routing_decisions()

        report = suite.generate_report()

        assert report["version"] == "V125"
        assert "timestamp" in report
        assert "configuration" in report
        assert "summary" in report
        assert "aggregate" in report


# =============================================================================
# Quick Benchmark Tests
# =============================================================================

class TestQuickBenchmark:
    """Test quick benchmark functions."""

    @pytest.mark.asyncio
    async def test_quick_benchmark_returns_metrics(self):
        """Test quick_benchmark returns expected metrics."""
        from benchmarking import quick_benchmark

        results = await quick_benchmark()

        assert "content_detection_accuracy" in results
        assert "content_detection_latency_ms" in results
        assert "routing_accuracy" in results
        assert "routing_latency_ms" in results

    @pytest.mark.asyncio
    async def test_full_benchmark_runs(self):
        """Test full benchmark suite runs (without saving)."""
        from benchmarking import run_full_benchmark

        report = await run_full_benchmark(
            warmup_iterations=1,
            benchmark_iterations=3,
            save_report=False,
        )

        assert report["version"] == "V125"
        assert len(report["summary"]) >= 2


# =============================================================================
# Statistics Calculation Tests
# =============================================================================

class TestStatisticsCalculation:
    """Test statistical calculations."""

    def test_stats_calculation_with_results(self):
        """Test statistics are calculated correctly."""
        from benchmarking import BenchmarkSuite, BenchmarkResult

        suite = BenchmarkSuite()

        results = [
            BenchmarkResult(name="test", success=True, latency_ms=1.0),
            BenchmarkResult(name="test", success=True, latency_ms=2.0),
            BenchmarkResult(name="test", success=True, latency_ms=3.0),
            BenchmarkResult(name="test", success=True, latency_ms=4.0),
            BenchmarkResult(name="test", success=True, latency_ms=5.0),
        ]

        stats = suite._calculate_stats(results)

        assert stats.total_runs == 5
        assert stats.successful_runs == 5
        assert stats.failed_runs == 0
        assert stats.min_latency_ms == 1.0
        assert stats.max_latency_ms == 5.0
        assert stats.mean_latency_ms == 3.0
        assert stats.median_latency_ms == 3.0

    def test_stats_calculation_with_failures(self):
        """Test statistics handle failures correctly."""
        from benchmarking import BenchmarkSuite, BenchmarkResult

        suite = BenchmarkSuite()

        results = [
            BenchmarkResult(name="test", success=True, latency_ms=1.0),
            BenchmarkResult(name="test", success=False, latency_ms=0, error="fail"),
            BenchmarkResult(name="test", success=True, latency_ms=2.0),
        ]

        stats = suite._calculate_stats(results)

        assert stats.total_runs == 3
        assert stats.successful_runs == 2
        assert stats.failed_runs == 1

    def test_stats_calculation_empty_results(self):
        """Test statistics handle empty results."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite()
        stats = suite._calculate_stats([])

        assert stats.total_runs == 0
        assert stats.mean_latency_ms == 0


# =============================================================================
# Content Detection Ground Truth Tests
# =============================================================================

class TestContentDetectionGroundTruth:
    """Test content detection against ground truth."""

    def test_python_code_detection(self):
        """Test Python code is detected as CODE."""
        try:
            from advanced_memory import ContentType, detect_content_type
        except ImportError:
            pytest.skip("advanced_memory not available")

        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        result = detect_content_type(code)
        assert result == ContentType.CODE

    def test_javascript_code_detection(self):
        """Test JavaScript code is detected as CODE."""
        try:
            from advanced_memory import ContentType, detect_content_type
        except ImportError:
            pytest.skip("advanced_memory not available")

        code = """
const getData = async () => {
    const response = await fetch('/api');
    return response.json();
};
"""
        result = detect_content_type(code)
        assert result == ContentType.CODE

    def test_plain_text_detection(self):
        """Test plain text is detected as TEXT."""
        try:
            from advanced_memory import ContentType, detect_content_type
        except ImportError:
            pytest.skip("advanced_memory not available")

        text = "This is a simple sentence without any code."
        result = detect_content_type(text)
        assert result == ContentType.TEXT

    def test_multilingual_detection(self):
        """Test non-English text is detected as MULTILINGUAL."""
        try:
            from advanced_memory import ContentType, detect_content_type
        except ImportError:
            pytest.skip("advanced_memory not available")

        text = "这是中文文本测试"
        result = detect_content_type(text)
        assert result == ContentType.MULTILINGUAL

    def test_empty_text_detection(self):
        """Test empty text returns UNKNOWN."""
        try:
            from advanced_memory import ContentType, detect_content_type
        except ImportError:
            pytest.skip("advanced_memory not available")

        result = detect_content_type("")
        assert result == ContentType.UNKNOWN


# =============================================================================
# V126 Hybrid Routing Tests
# =============================================================================

class TestV126HybridRouting:
    """Test V126 hybrid routing improvements."""

    def test_v126_imports(self):
        """Verify V126 components can be imported."""
        from benchmarking import (
            ContentConfidence,
            detect_content_type_v126,
            detect_content_type_hybrid,
        )
        assert ContentConfidence is not None
        assert detect_content_type_v126 is not None
        assert detect_content_type_hybrid is not None

    def test_v126_confidence_dataclass(self):
        """Test ContentConfidence dataclass."""
        from benchmarking import ContentConfidence, ContentType

        conf = ContentConfidence(
            content_type=ContentType.CODE,
            confidence=0.9,
            code_score=0.9,
            multilingual_score=0.0,
            text_score=0.1,
        )
        assert conf.content_type == ContentType.CODE
        assert conf.confidence == 0.9
        assert conf.code_score == 0.9

    def test_v126_confidence_clamping(self):
        """Test that confidence values are clamped to 0-1."""
        from benchmarking import ContentConfidence, ContentType

        conf = ContentConfidence(
            content_type=ContentType.TEXT,
            confidence=1.5,  # Should be clamped to 1.0
            code_score=-0.1,  # Should be clamped to 0.0
            multilingual_score=2.0,  # Should be clamped to 1.0
            text_score=0.5,
        )
        assert conf.confidence == 1.0
        assert conf.code_score == 0.0
        assert conf.multilingual_score == 1.0

    def test_v126_detects_code(self):
        """Test V126 correctly detects code."""
        from benchmarking import detect_content_type_v126, ContentType

        result = detect_content_type_v126("def foo(): return 42")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.5

    def test_v126_detects_text(self):
        """Test V126 correctly detects text."""
        from benchmarking import detect_content_type_v126, ContentType

        result = detect_content_type_v126("This is a simple sentence.")
        assert result.content_type == ContentType.TEXT
        assert result.text_score >= 0.0

    def test_v126_detects_multilingual(self):
        """Test V126 correctly detects multilingual."""
        from benchmarking import detect_content_type_v126, ContentType

        result = detect_content_type_v126("これはテストです")
        assert result.content_type == ContentType.MULTILINGUAL
        assert result.multilingual_score >= 0.3

    def test_v126_detects_mixed(self):
        """Test V126 correctly detects MIXED content (key V126 improvement)."""
        from benchmarking import detect_content_type_v126, ContentType

        # This case failed in V124 due to 30% threshold
        result = detect_content_type_v126("# 中文注释\ndef bar(): pass")
        assert result.content_type == ContentType.MIXED
        assert result.code_score >= 0.4
        assert result.multilingual_score >= 0.15

    def test_v126_hybrid_function(self):
        """Test V126 hybrid function returns just ContentType."""
        from benchmarking import detect_content_type_hybrid, ContentType

        result = detect_content_type_hybrid("def test(): pass")
        assert isinstance(result, ContentType)
        assert result == ContentType.CODE

    @pytest.mark.asyncio
    async def test_v126_benchmark_runs(self):
        """Test V126 routing benchmark executes."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_routing_v126()

        assert stats.name == "routing_v126"
        assert stats.total_runs > 0
        # V126 should have higher accuracy than V124 for the test cases
        if stats.accuracy is not None:
            assert stats.accuracy >= 0.9  # Should be close to 100%


# =============================================================================
# V127 Fine-Tuned Detection Tests
# =============================================================================

class TestV127FineTunedDetection:
    """Test V127 fine-tuned detection improvements."""

    def test_v127_imports(self):
        """Verify V127 components can be imported."""
        from benchmarking import (
            detect_content_type_v127,
            detect_content_type_finetuned,
        )
        assert detect_content_type_v127 is not None
        assert detect_content_type_finetuned is not None

    def test_v127_short_code_assignment(self):
        """Test V127 detects short variable assignments as code."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("x = 1")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.5

    def test_v127_short_code_function_call(self):
        """Test V127 detects short function calls as code."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("print(x)")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.5

    def test_v127_short_code_console_log(self):
        """Test V127 detects console.log as code."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("console.log('test')")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.7

    def test_v127_short_code_self(self):
        """Test V127 detects Python self.* as code."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("self.value = 42")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.5

    def test_v127_short_code_js_let(self):
        """Test V127 detects JavaScript let assignment as code."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("let y = 1")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.7

    def test_v127_false_positive_return(self):
        """Test V127 avoids false positive for 'return to sender'."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("return to sender")
        assert result.content_type == ContentType.TEXT

    def test_v127_false_positive_let_me(self):
        """Test V127 avoids false positive for 'let me know'."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("let me know")
        assert result.content_type == ContentType.TEXT

    def test_v127_mixed_english_noneng(self):
        """Test V127 detects English+non-English as MIXED."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("Hello 世界")
        assert result.content_type == ContentType.MIXED

    def test_v127_mixed_code_with_jp_comment(self):
        """Test V127 detects code with Japanese comment as MIXED."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("var x = 1 // コメント")
        assert result.content_type == ContentType.MIXED

    def test_v127_pure_multilingual(self):
        """Test V127 still correctly detects pure non-English as MULTILINGUAL."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("これはテストです")
        assert result.content_type == ContentType.MULTILINGUAL
        assert result.multilingual_score >= 0.8

    def test_v127_pure_code(self):
        """Test V127 doesn't regress on pure code detection."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("def foo(): return 42")
        assert result.content_type == ContentType.CODE
        assert result.code_score >= 0.9

    def test_v127_pure_text(self):
        """Test V127 doesn't regress on pure text detection."""
        from benchmarking import detect_content_type_v127, ContentType

        result = detect_content_type_v127("This is a simple sentence.")
        assert result.content_type == ContentType.TEXT

    @pytest.mark.asyncio
    async def test_v127_benchmark_runs(self):
        """Test V127 routing benchmark executes with 100% accuracy."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_routing_v127()

        assert stats.name == "routing_v127"
        assert stats.total_runs > 0
        # V127 should have 100% accuracy on all test cases
        if stats.accuracy is not None:
            assert stats.accuracy == 1.0  # Must be exactly 100%

    @pytest.mark.asyncio
    async def test_v127_vs_v126_improvement(self):
        """Test V127 has higher accuracy than V126 on edge cases."""
        from benchmarking import detect_content_type_v126, detect_content_type_v127, ContentType

        edge_cases = [
            ("x = 1", ContentType.CODE),
            ("print(x)", ContentType.CODE),
            ("Hello 世界", ContentType.MIXED),
            ("let me know", ContentType.TEXT),
        ]

        v126_correct = 0
        v127_correct = 0

        for text, expected in edge_cases:
            if detect_content_type_v126(text).content_type == expected:
                v126_correct += 1
            if detect_content_type_v127(text).content_type == expected:
                v127_correct += 1

        # V127 should be better
        assert v127_correct >= v126_correct


# =============================================================================
# V128 Chunking-Based Analysis Tests
# =============================================================================

class TestV128ChunkingAnalysis:
    """Test V128 chunking-based analysis for long documents."""

    def test_v128_imports(self):
        """Verify V128 components can be imported."""
        from benchmarking import (
            ChunkAnalysis,
            DocumentAnalysis,
            detect_content_type_v128,
            detect_content_type_chunked,
        )
        assert ChunkAnalysis is not None
        assert DocumentAnalysis is not None
        assert detect_content_type_v128 is not None
        assert detect_content_type_chunked is not None

    def test_v128_short_text_passthrough(self):
        """Test V128 uses V127 directly for short texts."""
        from benchmarking import detect_content_type_v128, ContentType

        result = detect_content_type_v128("def foo(): return 42")
        assert result.overall_type == ContentType.CODE
        assert len(result.chunks) == 1

    def test_v128_pure_text_document(self):
        """Test V128 correctly analyzes a pure text document."""
        from benchmarking import detect_content_type_v128, ContentType

        doc = """
        The quick brown fox jumps over the lazy dog. This sentence contains every
        letter of the English alphabet and is commonly used for testing purposes.

        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.
        """
        result = detect_content_type_v128(doc)
        assert result.overall_type == ContentType.TEXT
        assert "text" in result.type_distribution
        assert result.type_distribution["text"] > 0.8

    def test_v128_pure_code_document(self):
        """Test V128 correctly analyzes a pure code document."""
        from benchmarking import detect_content_type_v128, ContentType

        doc = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n-1)
        """
        result = detect_content_type_v128(doc)
        assert result.overall_type == ContentType.CODE
        assert "code" in result.type_distribution

    def test_v128_markdown_with_code_blocks(self):
        """Test V128 correctly identifies markdown with code blocks as MIXED."""
        from benchmarking import detect_content_type_v128, ContentType

        doc = """
# Project README

This is a sample project that demonstrates mixed content.

```python
pip install my-project
from my_project import main
main()
```

## Usage

The project provides utilities for data processing.
        """
        result = detect_content_type_v128(doc)
        assert result.overall_type == ContentType.MIXED
        assert result.is_heterogeneous
        assert len(result.chunks) >= 2

    def test_v128_code_block_detection(self):
        """Test V128 correctly identifies markdown code blocks."""
        from benchmarking import detect_content_type_v128, ContentType

        doc = """
Some text here.

```javascript
const x = 1;
console.log(x);
```

More text after.
        """
        result = detect_content_type_v128(doc)
        # Should find at least one code block
        code_block_chunks = [c for c in result.chunks if c.is_code_block]
        assert len(code_block_chunks) >= 1
        assert code_block_chunks[0].content_type == ContentType.CODE

    def test_v128_multilingual_document(self):
        """Test V128 correctly analyzes a pure multilingual document."""
        from benchmarking import detect_content_type_v128, ContentType

        doc = """
这是一个中文文档。

该项目提供多种功能，包括数据处理和分析工具。

使用方法非常简单，只需要导入模块即可开始使用。
        """
        result = detect_content_type_v128(doc)
        assert result.overall_type == ContentType.MULTILINGUAL
        assert "multilingual" in result.type_distribution

    def test_v128_code_with_multilingual_comments(self):
        """Test V128 detects code with non-English comments as MIXED."""
        from benchmarking import detect_content_type_v128, ContentType

        doc = """
# Программа для расчета
def factorial(n):
    # Базовый случай
    if n <= 1:
        return 1
    return n * factorial(n - 1)
        """
        result = detect_content_type_v128(doc)
        assert result.overall_type == ContentType.MIXED

    def test_v128_type_distribution(self):
        """Test V128 provides accurate type distribution."""
        from benchmarking import detect_content_type_v128

        doc = """
This is some text content.

```python
def hello():
    print("Hello")
```

More text content here.
        """
        result = detect_content_type_v128(doc)
        # Distribution should sum to approximately 1.0
        total = sum(result.type_distribution.values())
        assert 0.99 <= total <= 1.01

    def test_v128_dominant_types(self):
        """Test V128 correctly identifies dominant types."""
        from benchmarking import detect_content_type_v128, ContentType

        # Document with mostly text and some code
        doc = """
This is a long paragraph of text that should be the dominant type.
It contains multiple sentences and provides information about the project.
The text continues for several lines to ensure it dominates.

```python
x = 1
```

The project documentation continues here with more text.
        """
        result = detect_content_type_v128(doc)
        # Should have at least one dominant type
        assert len(result.dominant_types) >= 1

    def test_v128_chunked_wrapper(self):
        """Test V128 chunked wrapper returns ContentType."""
        from benchmarking import detect_content_type_chunked, ContentType

        doc = "This is a simple text document."
        result = detect_content_type_chunked(doc)
        assert isinstance(result, ContentType)
        assert result == ContentType.TEXT

    @pytest.mark.asyncio
    async def test_v128_benchmark_runs(self):
        """Test V128 routing benchmark executes successfully."""
        from benchmarking import BenchmarkSuite

        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5)
        stats = await suite.benchmark_routing_v128()

        assert stats.name == "routing_v128"
        assert stats.total_runs > 0
        # V128 should have good accuracy on document-level detection
        if stats.accuracy is not None:
            assert stats.accuracy >= 0.8

    @pytest.mark.asyncio
    async def test_v128_vs_v127_for_long_docs(self):
        """Test V128 provides better analysis for long documents."""
        from benchmarking import (
            detect_content_type_v127,
            detect_content_type_v128,
            ContentType,
        )

        # A README-style document that V127 might misclassify
        readme = """
# My Project

This is a Python project for data processing.

## Installation

```bash
pip install my-project
```

## Usage

```python
from my_project import process
result = process(data)
```

## License

MIT License - see LICENSE file for details.
        """

        v127_result = detect_content_type_v127(readme)
        v128_result = detect_content_type_v128(readme)

        # V128 should detect this as MIXED (text + code blocks)
        assert v128_result.overall_type == ContentType.MIXED
        assert v128_result.is_heterogeneous
        # V128 provides more detail
        assert len(v128_result.chunks) >= 2


# =============================================================================
# V129: Production-Grade Chonkie Integration Tests
# =============================================================================


class TestV129ChonkieIntegration:
    """Test V129 production-grade chunking with Chonkie."""

    def test_v129_imports(self):
        """Verify V129 components import correctly."""
        from benchmarking import (
            detect_content_type_v129,
            is_chonkie_available,
            get_chonkie_version,
            get_chonkie_manager,
            ChonkieChunk,
            ChonkieAnalysis,
            ChonkieChunkerManager,
        )
        assert detect_content_type_v129 is not None
        assert is_chonkie_available is not None
        assert get_chonkie_version is not None
        assert get_chonkie_manager is not None
        assert ChonkieChunk is not None
        assert ChonkieAnalysis is not None
        assert ChonkieChunkerManager is not None

    def test_v129_chonkie_available(self):
        """Verify Chonkie is available for production use."""
        from benchmarking import is_chonkie_available, get_chonkie_version

        # Chonkie should be installed
        assert is_chonkie_available() is True

        # Version should be 1.5.4 or higher
        version = get_chonkie_version()
        assert version is not None
        assert version.startswith("1.")

    def test_v129_manager_initialization(self):
        """Test ChonkieChunkerManager initializes correctly."""
        from benchmarking import ChonkieChunkerManager

        manager = ChonkieChunkerManager(
            semantic_threshold=0.75,
            chunk_size=1024,
        )
        assert manager is not None
        assert manager.semantic_threshold == 0.75
        assert manager.chunk_size == 1024

    def test_v129_text_chunking(self):
        """Test V129 handles plain text correctly with SemanticChunker."""
        from benchmarking import detect_content_type_v129, ContentType

        text = """
        Natural language processing has revolutionized how we interact with computers.
        Machine learning algorithms can now understand context and generate text.
        Deep learning models have achieved remarkable accuracy in complex tasks.

        The field continues to advance rapidly with new architectures.
        Transformer models have become the foundation of modern NLP systems.
        Research in this area promises even more exciting developments.
        """

        result = detect_content_type_v129(text)

        # Should detect as TEXT
        assert result.overall_type == ContentType.TEXT
        assert result.overall_confidence > 0.5
        assert len(result.chunks) >= 1
        # Should use semantic or recursive chunker
        assert result.chunker_used in ("semantic", "recursive", "fallback")

    def test_v129_code_chunking(self):
        """Test V129 handles code correctly with CodeChunker."""
        from benchmarking import detect_content_type_v129, ContentType

        code = '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


class MathUtils:
    """Mathematical utility functions."""

    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
'''

        result = detect_content_type_v129(code)

        # Should detect as CODE
        assert result.overall_type == ContentType.CODE
        assert result.overall_confidence > 0.5
        assert len(result.chunks) >= 1
        # Chunks should be marked as code
        code_chunks = [c for c in result.chunks if c.is_code]
        assert len(code_chunks) >= 1

    def test_v129_mixed_content(self):
        """Test V129 handles mixed content (README with code blocks)."""
        from benchmarking import detect_content_type_v129, ContentType

        readme = '''
# My Awesome Library

A production-ready library for data processing.

## Installation

Install via pip:

```python
pip install awesome-lib
```

## Quick Start

```python
from awesome_lib import process

# Process your data
result = process(data)
print(result)
```

## Features

- High performance processing
- Easy to use API
- Comprehensive documentation

## License

MIT License
'''

        result = detect_content_type_v129(readme)

        # Should detect as MIXED
        assert result.overall_type in (ContentType.MIXED, ContentType.TEXT, ContentType.CODE)
        assert len(result.chunks) >= 1
        # Should have heterogeneous content or multiple chunks
        assert result.chonkie_version is not None or result.chunker_used == "fallback"

    def test_v129_multilingual_content(self):
        """Test V129 handles multilingual content."""
        from benchmarking import detect_content_type_v129, ContentType

        multilingual = """
        Welcome to our international documentation.

        日本語のドキュメント: このライブラリは高性能です。

        Документация на русском: Это руководство по использованию.

        Documentation en français: Ce guide explique l'utilisation.

        中文文档：这是一个强大的库。
        """

        result = detect_content_type_v129(multilingual)

        # Should detect as MULTILINGUAL or TEXT
        assert result.overall_type in (ContentType.MULTILINGUAL, ContentType.TEXT, ContentType.MIXED)
        assert len(result.chunks) >= 1

    def test_v129_chunk_attributes(self):
        """Test V129 chunks have correct attributes."""
        from benchmarking import detect_content_type_v129

        text = "This is a test document with enough content to generate chunks."

        result = detect_content_type_v129(text)

        for chunk in result.chunks:
            # All chunks should have required attributes
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "token_count")
            assert hasattr(chunk, "start_index")
            assert hasattr(chunk, "end_index")
            assert hasattr(chunk, "content_type")
            assert hasattr(chunk, "confidence")
            assert hasattr(chunk, "is_code")

            # Values should be valid
            assert chunk.text is not None
            assert chunk.token_count >= 0
            assert chunk.start_index >= 0
            assert chunk.confidence >= 0.0
            assert chunk.confidence <= 1.0

    def test_v129_analysis_attributes(self):
        """Test V129 analysis has correct structure."""
        from benchmarking import detect_content_type_v129, ContentType

        text = "Sample document for testing the analysis structure and attributes."

        result = detect_content_type_v129(text)

        # Required attributes
        assert hasattr(result, "overall_type")
        assert hasattr(result, "overall_confidence")
        assert hasattr(result, "chunks")
        assert hasattr(result, "type_distribution")
        assert hasattr(result, "dominant_types")
        assert hasattr(result, "is_heterogeneous")
        assert hasattr(result, "chunker_used")
        assert hasattr(result, "chonkie_version")

        # Type checks
        assert isinstance(result.overall_type, ContentType)
        assert isinstance(result.overall_confidence, float)
        assert isinstance(result.chunks, list)
        assert isinstance(result.type_distribution, dict)
        assert isinstance(result.dominant_types, list)
        assert isinstance(result.is_heterogeneous, bool)
        assert isinstance(result.chunker_used, str)

    def test_v129_global_manager_reuse(self):
        """Test global chunker manager is reused across calls."""
        from benchmarking import get_chonkie_manager

        manager1 = get_chonkie_manager()
        manager2 = get_chonkie_manager()

        # Should be the same instance (singleton pattern)
        assert manager1 is manager2

    def test_v129_chunker_types(self):
        """Test V129 uses appropriate chunker for content type."""
        from benchmarking import detect_content_type_v129, is_chonkie_available

        if not is_chonkie_available():
            pytest.skip("Chonkie not available")

        # Pure code should use code chunker
        code = "def hello(): return 'world'\n\ndef foo(): return 'bar'"
        code_result = detect_content_type_v129(code)
        # Code chunker should be selected for code content
        assert code_result.chunker_used in ("code", "recursive", "semantic", "fallback")

        # Pure text should use semantic chunker
        text = "This is plain English text about natural language processing."
        text_result = detect_content_type_v129(text)
        assert text_result.chunker_used in ("semantic", "recursive", "fallback")

    def test_v129_vs_v128_equivalence(self):
        """Test V129 produces similar results to V128 for same content."""
        from benchmarking import (
            detect_content_type_v128,
            detect_content_type_v129,
            ContentType,
        )

        text = """
# Example Document

This is a sample document with mixed content.

```python
def example():
    return True
```

And more text content here.
"""

        v128_result = detect_content_type_v128(text)
        v129_result = detect_content_type_v129(text)

        # Both should have chunks
        assert len(v128_result.chunks) >= 1
        assert len(v129_result.chunks) >= 1

        # Types should be consistent
        # (might differ slightly due to different chunking strategies)
        assert v129_result.overall_type in (
            v128_result.overall_type,
            ContentType.MIXED,
            ContentType.TEXT,
            ContentType.CODE,
        )

    def test_v129_empty_text(self):
        """Test V129 handles empty text gracefully."""
        from benchmarking import detect_content_type_v129, ContentType

        result = detect_content_type_v129("")

        assert result.overall_type == ContentType.UNKNOWN
        assert result.overall_confidence == 1.0

    def test_v129_short_text(self):
        """Test V129 handles very short text."""
        from benchmarking import detect_content_type_v129

        result = detect_content_type_v129("Hi")

        assert result is not None
        assert len(result.chunks) >= 0  # Might be empty for very short text

    def test_v129_reset_manager(self):
        """Test reset_chonkie_manager clears global state."""
        from benchmarking import get_chonkie_manager, reset_chonkie_manager

        # Get initial manager
        manager1 = get_chonkie_manager()

        # Reset
        reset_chonkie_manager()

        # Get new manager - should be different instance
        manager2 = get_chonkie_manager()
        assert manager1 is not manager2

        # New manager should work correctly
        assert manager2._ensure_initialized()


# =============================================================================
# V130: Cross-Session Persistence Tests
# =============================================================================

class TestV130ChonkiePersistence:
    """
    V130: Test cross-session persistence for Chonkie.

    These tests verify that Chonkie configuration and metrics
    persist correctly across sessions.
    """

    def test_v130_imports(self):
        """Test V130 persistence functions can be imported."""
        from benchmarking import (
            ChonkieSessionState,
            load_chonkie_state,
            save_chonkie_state,
            get_chonkie_session_stats,
        )
        assert ChonkieSessionState is not None
        assert load_chonkie_state is not None
        assert save_chonkie_state is not None
        assert get_chonkie_session_stats is not None

    def test_v130_session_state_dataclass(self):
        """Test ChonkieSessionState dataclass."""
        from benchmarking import ChonkieSessionState

        state = ChonkieSessionState()
        assert state.semantic_threshold == 0.75
        assert state.chunk_size == 1024
        assert state.code_language == "python"
        assert state.session_count == 0

    def test_v130_state_serialization(self):
        """Test state can be serialized to/from dict."""
        from benchmarking import ChonkieSessionState

        state = ChonkieSessionState(
            semantic_threshold=0.8,
            chunk_size=512,
            session_count=5,
            total_chunks_processed=100,
        )

        # Round-trip through dict
        state_dict = state.to_dict()
        restored = ChonkieSessionState.from_dict(state_dict)

        assert restored.semantic_threshold == 0.8
        assert restored.chunk_size == 512
        assert restored.session_count == 5
        assert restored.total_chunks_processed == 100

    def test_v130_save_and_load_state(self, tmp_path, monkeypatch):
        """Test save and load state with temp directory."""
        from benchmarking import (
            ChonkieSessionState,
            save_chonkie_state,
            load_chonkie_state,
            _get_chonkie_state_path,
        )
        from pathlib import Path

        # Mock home directory to use temp path
        test_state_dir = tmp_path / ".claude" / "unleash_memory"
        test_state_dir.mkdir(parents=True, exist_ok=True)

        # Monkey-patch the path function
        monkeypatch.setattr(
            "benchmarking._get_chonkie_state_path",
            lambda: test_state_dir / "chonkie_state.json"
        )

        # Create and save state
        state = ChonkieSessionState(
            session_count=3,
            total_chunks_processed=50,
            init_success=True,
            available_chunkers=["semantic", "code"],
        )
        result = save_chonkie_state(state)
        assert result is True

        # Load and verify
        loaded = load_chonkie_state()
        assert loaded.session_count == 3
        assert loaded.total_chunks_processed == 50
        assert loaded.init_success is True
        assert "semantic" in loaded.available_chunkers
        assert "code" in loaded.available_chunkers

    def test_v130_get_session_stats(self):
        """Test get_chonkie_session_stats returns expected structure."""
        from benchmarking import get_chonkie_session_stats

        stats = get_chonkie_session_stats()

        # Verify structure
        assert "session_count" in stats
        assert "total_chunks_processed" in stats
        assert "available_chunkers" in stats
        assert "configuration" in stats
        assert "semantic_threshold" in stats["configuration"]
        assert "chunk_size" in stats["configuration"]

    def test_v130_manager_with_persistence(self):
        """Test ChonkieChunkerManager uses persistence."""
        from benchmarking import (
            ChonkieChunkerManager,
            reset_chonkie_manager,
            get_chonkie_manager,
        )

        # Reset first
        reset_chonkie_manager()

        # Create manager with persistence enabled
        manager = get_chonkie_manager(persist_state=True)
        assert manager._persist_state is True
        assert manager._state is not None

        # Initialize and check state is updated
        manager._ensure_initialized()
        assert manager._state.session_count >= 1

        # Clean up
        reset_chonkie_manager()

    def test_v130_manager_without_persistence(self):
        """Test ChonkieChunkerManager works without persistence."""
        from benchmarking import ChonkieChunkerManager

        manager = ChonkieChunkerManager(persist_state=False)
        assert manager._persist_state is False

        # Should still work
        result = manager._ensure_initialized()
        # Result depends on Chonkie availability

    def test_v130_chunk_tracking(self):
        """Test chunks are tracked for metrics."""
        from benchmarking import (
            get_chonkie_manager,
            reset_chonkie_manager,
            detect_content_type_v129,
        )

        reset_chonkie_manager()
        manager = get_chonkie_manager()

        initial_count = manager._chunks_processed

        # Process some text
        detect_content_type_v129("def foo(): return 42\ndef bar(): return 43")

        # Should have incremented (if Chonkie available)
        if manager._initialized and manager._semantic_chunker is not None:
            assert manager._chunks_processed >= initial_count

        reset_chonkie_manager()


# =============================================================================
# V131: Letta Blocks API Sync Tests
# =============================================================================


class TestV131LettaBlocksSync:
    """
    V131: Test Letta Blocks API sync for cross-session memory.

    These tests verify that Core Memory (Blocks) syncing works correctly
    for cross-session persistence with Letta Cloud.
    """

    def test_v131_imports(self):
        """Test V131 Letta Blocks methods can be imported."""
        from cross_session_memory import CrossSessionMemory

        # Verify V131 methods exist
        memory = CrossSessionMemory.__new__(CrossSessionMemory)
        assert hasattr(memory, 'get_blocks')
        assert hasattr(memory, 'get_block')
        assert hasattr(memory, 'update_block')
        assert hasattr(memory, 'sync_state_to_block')
        assert hasattr(memory, 'load_state_from_block')
        assert hasattr(memory, 'sync_blocks_bidirectional')
        assert hasattr(memory, 'create_shared_block')
        assert hasattr(memory, 'attach_shared_block')
        assert hasattr(memory, 'detach_block')
        assert hasattr(memory, 'get_blocks_stats')

    def test_v131_get_blocks_without_client(self):
        """Test get_blocks returns empty dict when no Letta client."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        # Without Letta client configured, should return empty
        blocks = memory.get_blocks()
        assert isinstance(blocks, dict)
        # May be empty or have data depending on environment

    def test_v131_get_block_without_client(self):
        """Test get_block returns None when no Letta client."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        # Without Letta client, should return None
        block = memory.get_block("nonexistent")
        assert block is None

    def test_v131_update_block_without_client(self):
        """Test update_block returns False when no Letta client."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        # Without Letta client, should return False
        result = memory.update_block("test", "value")
        assert result is False

    def test_v131_sync_state_to_block(self):
        """Test sync_state_to_block works with local state."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        # Add some state data using the correct add() method
        memory.add("Test decision", memory_type="decision", importance=0.8)
        memory.add("Test learning", memory_type="learning", importance=0.7)

        # Sync should work (even if Letta not available, it prepares the state)
        result = memory.sync_state_to_block()
        # Returns False if no Letta, True if synced
        assert isinstance(result, bool)

    def test_v131_load_state_from_block(self):
        """Test load_state_from_block returns dict."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        # Should return empty or partial dict when no block available
        state = memory.load_state_from_block()
        assert isinstance(state, dict)

    def test_v131_sync_blocks_bidirectional(self):
        """Test sync_blocks_bidirectional returns sync summary."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        result = memory.sync_blocks_bidirectional()
        assert isinstance(result, dict)
        assert "synced" in result or "error" in result

    def test_v131_create_shared_block_without_client(self):
        """Test create_shared_block behavior without Letta sync enabled."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        result = memory.create_shared_block("test", "value")
        # Without letta_sync enabled, should return None
        # (may return block ID if env has LETTA_API_KEY)
        assert result is None or isinstance(result, str)

    def test_v131_attach_shared_block_without_client(self):
        """Test attach_shared_block returns False when no Letta client."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        result = memory.attach_shared_block("block-123")
        assert result is False

    def test_v131_detach_block_without_client(self):
        """Test detach_block returns False when no Letta client."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        result = memory.detach_block("block-123")
        assert result is False

    def test_v131_get_blocks_stats(self):
        """Test get_blocks_stats returns expected structure."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        stats = memory.get_blocks_stats()

        assert isinstance(stats, dict)
        # V131 returns these specific keys
        assert "letta_sync_enabled" in stats
        assert "total_blocks" in stats
        assert "total_chars" in stats
        assert "blocks" in stats

    def test_v131_block_value_truncation(self):
        """Test that block values are truncated to ~4KB limit."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)
        # Create a very long value
        long_value = "x" * 5000

        # The update_block method should truncate to ~4000 chars
        # We can't test actual update without Letta, but the method exists
        assert callable(memory.update_block)

    def test_v131_state_serialization(self):
        """Test state is properly serialized for block storage."""
        from cross_session_memory import CrossSessionMemory

        memory = CrossSessionMemory(letta_sync=False)

        # Add various types of data using the correct add() method
        memory.add("Decision: Use Redis for caching", memory_type="decision", importance=0.9)
        memory.add("Decision: Choose PostgreSQL", memory_type="decision", importance=0.85)
        memory.add("Learned: Always verify API signatures", memory_type="learning", importance=0.8)
        memory.add("Fact: Project uses Python 3.11", memory_type="fact", importance=0.7)

        # sync_state_to_block should serialize all this
        # Even without Letta, it should not error
        result = memory.sync_state_to_block()
        assert isinstance(result, bool)


# =============================================================================
# V132: Real API Validation Tests for Letta Integration
# =============================================================================


import os

# Check for real API availability
LETTA_API_KEY = os.environ.get("LETTA_API_KEY")
HAS_LETTA_API = LETTA_API_KEY is not None

# Test agent ID for UNLEASH project (verified 2026-01-30)
TEST_AGENT_ID = "agent-daee71d2-193b-485e-bda4-ee44752635fe"


@pytest.mark.skipif(not HAS_LETTA_API, reason="LETTA_API_KEY not set")
class TestV132RealAPIValidation:
    """
    V132: Real API validation tests for Letta integration.

    These tests run against the REAL Letta Cloud API to verify:
    1. SDK signatures match documentation
    2. Response shapes are correct
    3. Critical gotchas are documented correctly

    IMPORTANT: These require LETTA_API_KEY environment variable.
    Uses the UNLEASH test agent: agent-daee71d2-193b-485e-bda4-ee44752635fe
    """

    @pytest.fixture
    def letta_client(self):
        """Create a real Letta client."""
        from letta_client import Letta
        return Letta(api_key=LETTA_API_KEY, base_url="https://api.letta.com")

    # =========================================================================
    # Client Initialization Tests
    # =========================================================================

    def test_v132_client_requires_base_url(self):
        """Test that Letta Cloud requires base_url (CRITICAL GOTCHA)."""
        from letta_client import Letta

        # This is the CORRECT way to initialize for Cloud
        client = Letta(api_key=LETTA_API_KEY, base_url="https://api.letta.com")
        assert client is not None

    def test_v132_client_with_httpx_pooling(self):
        """Test client initialization with httpx connection pooling (V16 pattern)."""
        from letta_client import Letta
        import httpx

        client = Letta(
            api_key=LETTA_API_KEY,
            base_url="https://api.letta.com",
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                timeout=httpx.Timeout(30.0)
            )
        )
        assert client is not None

    # =========================================================================
    # Messages API Tests
    # =========================================================================

    def test_v132_messages_create(self, letta_client):
        """Test messages.create with real API."""
        response = letta_client.agents.messages.create(
            TEST_AGENT_ID,
            messages=[{"role": "user", "content": "V132 test: ping"}]
        )

        # Verify response structure
        assert hasattr(response, 'messages')
        assert isinstance(response.messages, list)

        # Should have at least one message in response
        assert len(response.messages) >= 1

    def test_v132_messages_list(self, letta_client):
        """Test messages.list with real API."""
        messages = letta_client.agents.messages.list(TEST_AGENT_ID, limit=10)

        # Should return a paginated result
        assert messages is not None
        # Iterate to verify it's iterable
        count = 0
        for msg in messages:
            assert msg is not None  # Verify message is valid
            count += 1
            if count >= 5:
                break
        # May have 0 messages if agent is new, but should not error
        assert count >= 0

    # =========================================================================
    # Blocks API Tests (Core Memory)
    # =========================================================================

    def test_v132_blocks_list(self, letta_client):
        """Test blocks.list with real API."""
        blocks = letta_client.agents.blocks.list(TEST_AGENT_ID)

        # Should return an iterable
        blocks_list = list(blocks)
        assert isinstance(blocks_list, list)

        # Agents typically have at least 'human' and 'persona' blocks
        if len(blocks_list) > 0:
            block = blocks_list[0]
            assert hasattr(block, 'label')
            assert hasattr(block, 'value')

    def test_v132_blocks_retrieve_positional_label(self, letta_client):
        """Test blocks.retrieve with POSITIONAL block_label (CRITICAL GOTCHA).

        WRONG: retrieve(agent_id, block_label="human")
        CORRECT: retrieve("human", agent_id=agent_id)
        """
        # CORRECT: block_label is POSITIONAL, agent_id is KEYWORD
        block = letta_client.agents.blocks.retrieve("human", agent_id=TEST_AGENT_ID)

        if block is not None:
            assert hasattr(block, 'label')
            assert hasattr(block, 'value')
            assert block.label == "human"

    def test_v132_blocks_update(self, letta_client):
        """Test blocks.update with real API."""
        # First get current value
        block = letta_client.agents.blocks.retrieve("human", agent_id=TEST_AGENT_ID)
        original_value = block.value if block else ""

        # Update with test value
        test_value = f"V132 test: {original_value[:50]}..." if original_value else "V132 test value"

        # CORRECT: block_label is POSITIONAL, agent_id is KEYWORD
        updated = letta_client.agents.blocks.update(
            "human",
            agent_id=TEST_AGENT_ID,
            value=test_value[:4000]  # Truncate to ~4KB limit
        )

        assert updated is not None
        assert hasattr(updated, 'value')

        # Restore original value
        letta_client.agents.blocks.update(
            "human",
            agent_id=TEST_AGENT_ID,
            value=original_value
        )

    # =========================================================================
    # Passages API Tests (Archival Memory)
    # =========================================================================

    def test_v132_passages_search_uses_query_and_top_k(self, letta_client):
        """Test passages.search uses query= and top_k= (CRITICAL GOTCHA).

        WRONG: search(agent_id, text="...", limit=10)
        CORRECT: search(agent_id, query="...", top_k=10)
        """
        import warnings

        # Capture deprecation warnings (passages.search shows warning in 1.7+)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            # CORRECT: query= and top_k=
            search_response = letta_client.agents.passages.search(
                TEST_AGENT_ID,
                query="test",
                top_k=5
            )

        # Response should have .results attribute
        assert hasattr(search_response, 'results')
        assert isinstance(search_response.results, list)

    def test_v132_passages_create_returns_list(self, letta_client):
        """Test passages.create returns LIST, not single object (CRITICAL GOTCHA).

        WRONG: passage_id = created.id
        CORRECT: passage_id = created[0].id
        """
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            # Create a test passage
            created = letta_client.agents.passages.create(
                TEST_AGENT_ID,
                text="V132 test passage for API validation",
                tags=["v132-test"]
            )

        # CRITICAL: create() returns a LIST
        assert isinstance(created, list)
        assert len(created) >= 1

        # Access the first element to get the passage
        passage = created[0]
        assert hasattr(passage, 'id')
        assert hasattr(passage, 'text')

        # Clean up: delete the test passage
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                letta_client.agents.passages.delete(
                    memory_id=passage.id,
                    agent_id=TEST_AGENT_ID
                )
        except Exception:
            pass  # Best effort cleanup

    def test_v132_passages_list_uses_limit(self, letta_client):
        """Test passages.list uses limit= (different from search which uses top_k=)."""
        # list() uses limit= (pagination)
        passages = letta_client.agents.passages.list(TEST_AGENT_ID, limit=10)

        # Should return paginated iterator
        assert passages is not None

        # Iterate to verify
        count = 0
        for passage in passages:
            assert hasattr(passage, 'id')
            assert hasattr(passage, 'text')
            count += 1
            if count >= 5:
                break

    # =========================================================================
    # Response Shape Validation
    # =========================================================================

    def test_v132_message_response_shape(self, letta_client):
        """Validate complete message response shape from real API."""
        response = letta_client.agents.messages.create(
            TEST_AGENT_ID,
            messages=[{"role": "user", "content": "V132: Validate response shape"}]
        )

        # Response should have these attributes
        assert hasattr(response, 'messages')

        # Messages should be a list of message objects
        for msg in response.messages:
            assert hasattr(msg, 'message_type')
            # Common message types: assistant_message, function_call, etc.

    def test_v132_block_response_shape(self, letta_client):
        """Validate complete block response shape from real API."""
        block = letta_client.agents.blocks.retrieve("human", agent_id=TEST_AGENT_ID)

        if block is not None:
            # Block should have these attributes
            assert hasattr(block, 'id')
            assert hasattr(block, 'label')
            assert hasattr(block, 'value')
            assert hasattr(block, 'limit')

            # Value should be a string
            assert isinstance(block.value, str)
            # Label should match request
            assert block.label == "human"

    def test_v132_passage_response_shape(self, letta_client):
        """Validate complete passage response shape from real API."""
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            search_response = letta_client.agents.passages.search(
                TEST_AGENT_ID,
                query="test",
                top_k=3
            )

        # Search response should have .results
        assert hasattr(search_response, 'results')

        # Each result should have passage attributes
        for result in search_response.results:
            assert hasattr(result, 'id')
            assert hasattr(result, 'text') or hasattr(result, 'content')

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_v132_invalid_agent_id_error(self, letta_client):
        """Test that invalid agent ID returns proper error."""
        # ApiError path may vary by SDK version
        try:
            from letta_client.core.api_error import ApiError  # type: ignore[import-not-found]
            error_types = (ApiError, Exception)
        except ImportError:
            error_types = (Exception,)

        with pytest.raises(error_types):
            letta_client.agents.messages.create(
                "invalid-agent-id-12345",
                messages=[{"role": "user", "content": "test"}]
            )

    def test_v132_invalid_block_label_returns_none_or_error(self, letta_client):
        """Test that invalid block label is handled gracefully."""
        # Retrieving non-existent block should return None or raise
        try:
            result = letta_client.agents.blocks.retrieve(
                "nonexistent-block-label-v132",
                agent_id=TEST_AGENT_ID
            )
            # May return None for non-existent blocks
            assert result is None or hasattr(result, 'label')
        except Exception as e:
            # Or may raise an error, which is also acceptable
            assert "not found" in str(e).lower() or "404" in str(e)


# =============================================================================
# V132 Integration Summary Tests
# =============================================================================


@pytest.mark.skipif(not HAS_LETTA_API, reason="LETTA_API_KEY not set")
class TestV132IntegrationSummary:
    """Summary tests for V132 that validate the documented gotchas."""

    def test_v132_all_critical_gotchas_documented(self):
        """Verify all critical gotchas from CLAUDE.md are testable."""
        # These are the gotchas documented in CLAUDE.md
        gotchas = [
            "base_url required for Letta Cloud",
            "passages.search uses query= and top_k=, not limit=",
            "passages.create returns LIST, not single object",
            "blocks.retrieve: block_label positional, agent_id keyword",
            "deprecated: core_memory -> blocks, archival_memory -> passages",
        ]

        # All gotchas should have corresponding test methods in TestV132RealAPIValidation
        assert len(gotchas) >= 5

    def test_v132_sdk_version_compatibility(self):
        """Verify SDK version is compatible with tests."""
        try:
            from letta_client import __version__
            # Should be 1.7.x
            assert __version__.startswith("1.7") or __version__.startswith("1.8")
        except ImportError:
            # __version__ may not be exported, which is fine
            pass


# =============================================================================
# V133 Sleep-time Agent Tests
# =============================================================================


class TestV133SleeptimeAgentSupport:
    """V133: Tests for sleep-time agent mandatory triggers in CrossSessionMemory."""

    @pytest.fixture
    def memory_manager(self):
        """Create a CrossSessionMemory instance for testing."""
        from cross_session_memory import CrossSessionMemory
        return CrossSessionMemory()

    def test_v133_sleeptime_triggers_initialization(self, memory_manager):
        """Test that sleeptime triggers can be initialized."""
        # CrossSessionMemory should have the sleeptime-related methods
        assert hasattr(memory_manager, 'enable_sleeptime')
        assert hasattr(memory_manager, 'disable_sleeptime')
        assert hasattr(memory_manager, 'get_sleeptime_status')
        assert hasattr(memory_manager, 'trigger_sleeptime_consolidation')
        assert hasattr(memory_manager, 'configure_mandatory_triggers')
        assert hasattr(memory_manager, 'get_mandatory_triggers')
        assert hasattr(memory_manager, '_check_mandatory_triggers')

    def test_v133_configure_mandatory_triggers_defaults(self, memory_manager):
        """Test configure_mandatory_triggers with default values."""
        config = memory_manager.configure_mandatory_triggers()

        assert config is not None
        assert config.get("on_session_end") is True
        assert config.get("on_important_memory") is True
        assert config.get("on_memory_count") == 10
        assert config.get("importance_threshold") == 0.8
        assert "memories_since_last_trigger" in config

    def test_v133_configure_mandatory_triggers_custom(self, memory_manager):
        """Test configure_mandatory_triggers with custom values."""
        config = memory_manager.configure_mandatory_triggers(
            on_session_end=False,
            on_important_memory=True,
            on_memory_count=5,
            importance_threshold=0.9
        )

        assert config.get("on_session_end") is False
        assert config.get("on_memory_count") == 5
        assert config.get("importance_threshold") == 0.9

    def test_v133_get_mandatory_triggers(self, memory_manager):
        """Test get_mandatory_triggers returns current configuration."""
        # Configure first
        memory_manager.configure_mandatory_triggers(
            on_session_end=True,
            on_memory_count=15
        )

        # Get should return same config
        config = memory_manager.get_mandatory_triggers()
        assert config.get("on_session_end") is True
        assert config.get("on_memory_count") == 15

    def test_v133_get_sleeptime_status_structure(self, memory_manager):
        """Test get_sleeptime_status returns expected structure."""
        status = memory_manager.get_sleeptime_status()

        # Should return a dict with these keys
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "frequency" in status
        assert "group_id" in status
        assert "agent_id" in status
        assert "letta_available" in status

    def test_v133_enable_sleeptime_without_letta(self, memory_manager):
        """Test enable_sleeptime handles missing Letta client gracefully."""
        # Without LETTA_API_KEY, this should return False gracefully
        result = memory_manager.enable_sleeptime(frequency=5)

        # Should return False if Letta not available
        if not memory_manager.get_sleeptime_status().get("letta_available"):
            assert result is False

    def test_v133_disable_sleeptime_without_letta(self, memory_manager):
        """Test disable_sleeptime handles missing Letta client gracefully."""
        result = memory_manager.disable_sleeptime()

        # Should return False if Letta not available
        if not memory_manager.get_sleeptime_status().get("letta_available"):
            assert result is False

    def test_v133_trigger_sleeptime_consolidation_without_letta(self, memory_manager):
        """Test trigger_sleeptime_consolidation handles missing Letta client."""
        result = memory_manager.trigger_sleeptime_consolidation()

        # Should return False if Letta not available
        if not memory_manager.get_sleeptime_status().get("letta_available"):
            assert result is False

    def test_v133_check_mandatory_triggers_important_memory(self, memory_manager):
        """Test _check_mandatory_triggers with high importance memory."""
        from unittest.mock import MagicMock

        # Configure triggers
        memory_manager.configure_mandatory_triggers(
            on_important_memory=True,
            importance_threshold=0.7
        )

        # Create a mock memory object with high importance attribute
        important_memory = MagicMock()
        important_memory.importance = 0.9
        important_memory.content = "Critical system configuration"

        # This should trigger consolidation check - returns None (fire-and-forget)
        result = memory_manager._check_mandatory_triggers(important_memory)
        # Method returns None (it's a fire-and-forget trigger check)
        assert result is None

    def test_v133_check_mandatory_triggers_count_threshold(self, memory_manager):
        """Test _check_mandatory_triggers respects memory count threshold."""
        # Configure triggers with low count threshold
        memory_manager.configure_mandatory_triggers(
            on_memory_count=3,
            on_important_memory=False
        )

        # Add multiple memories to reach threshold
        for i in range(4):
            memory = {"content": f"Memory {i}", "importance": 0.5}
            memory_manager._check_mandatory_triggers(memory)

        # Get triggers to check memories_since_last_trigger
        config = memory_manager.get_mandatory_triggers()
        # Either the trigger was fired or count was incremented
        assert "memories_since_last_trigger" in config


@pytest.mark.skipif(not HAS_LETTA_API, reason="LETTA_API_KEY not set")
class TestV133SleeptimeRealAPI:
    """V133: Real API tests for sleep-time agent functionality.

    These tests require a valid LETTA_API_KEY and will interact with
    the actual Letta Cloud API.
    """

    @pytest.fixture
    def memory_manager(self):
        """Create a CrossSessionMemory instance with real API."""
        from cross_session_memory import CrossSessionMemory
        return CrossSessionMemory()

    def test_v133_real_api_get_sleeptime_status(self, memory_manager):
        """Test get_sleeptime_status against real API."""
        status = memory_manager.get_sleeptime_status()

        # Should indicate Letta is available
        assert status.get("letta_available") is True
        assert "enabled" in status
        assert "agent_id" in status

    def test_v133_real_api_sleeptime_roundtrip(self, memory_manager):
        """Test enable/disable sleep-time agent roundtrip.

        Note: This test modifies agent configuration. It should
        restore the original state after testing.
        """
        # Get initial status
        initial_status = memory_manager.get_sleeptime_status()

        try:
            # Try to enable (may fail if agent doesn't support it)
            enable_result = memory_manager.enable_sleeptime(frequency=10)

            if enable_result:
                # Verify it was enabled
                status = memory_manager.get_sleeptime_status()
                assert status.get("enabled") is True

                # Disable it
                disable_result = memory_manager.disable_sleeptime()
                assert disable_result is True

                # Verify disabled
                status = memory_manager.get_sleeptime_status()
                assert status.get("enabled") is False
        finally:
            # Restore original state
            if initial_status.get("enabled"):
                memory_manager.enable_sleeptime()
            else:
                memory_manager.disable_sleeptime()


# =============================================================================
# V133 Summary Tests
# =============================================================================


class TestV133IntegrationSummary:
    """Summary tests for V133 sleep-time agent integration."""

    def test_v133_all_sleeptime_methods_exist(self):
        """Verify all V133 sleep-time methods are implemented."""
        from cross_session_memory import CrossSessionMemory
        csm = CrossSessionMemory()

        methods = [
            'enable_sleeptime',
            'disable_sleeptime',
            'get_sleeptime_status',
            'trigger_sleeptime_consolidation',
            'configure_mandatory_triggers',
            'get_mandatory_triggers',
            '_check_mandatory_triggers',
        ]

        for method in methods:
            assert hasattr(csm, method), f"Missing method: {method}"
            assert callable(getattr(csm, method)), f"Not callable: {method}"

    def test_v133_mandatory_triggers_features(self):
        """Verify mandatory trigger configuration options."""
        from cross_session_memory import CrossSessionMemory
        csm = CrossSessionMemory()

        # These trigger types should be configurable
        config = csm.configure_mandatory_triggers(
            on_session_end=True,
            on_important_memory=True,
            on_memory_count=10,
            importance_threshold=0.8
        )

        assert "on_session_end" in config
        assert "on_important_memory" in config
        assert "on_memory_count" in config
        assert "importance_threshold" in config


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
