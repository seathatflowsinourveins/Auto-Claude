"""
Research Module - Platform Core
===============================

Provides research quality utilities for the AI platform.

Modules:
    - finding_quality: GAP-02 resolution for findings quality issues

Usage:
    from core.research import (
        FindingQualityProcessor,
        FindingQualityConfig,
        process_exa_results,
        process_tavily_results,
        process_perplexity_results,
    )

    processor = FindingQualityProcessor()
    findings = processor.process_findings(raw_findings, source_texts)
"""

from core.research.finding_quality import (
    # Main processor
    FindingQualityProcessor,
    # Configuration
    FindingQualityConfig,
    # Scoring
    QualityScorer,
    QualityScore,
    # Extraction
    ContentExtractor,
    # Detection
    TitleDetector,
    GarbagePatterns,
    # Convenience functions
    process_exa_results,
    process_tavily_results,
    process_perplexity_results,
)

__all__ = [
    'FindingQualityProcessor',
    'FindingQualityConfig',
    'QualityScorer',
    'QualityScore',
    'ContentExtractor',
    'TitleDetector',
    'GarbagePatterns',
    'process_exa_results',
    'process_tavily_results',
    'process_perplexity_results',
]
