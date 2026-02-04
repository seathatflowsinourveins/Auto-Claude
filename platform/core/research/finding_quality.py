"""
Finding Quality Module - GAP-02 Resolution
==========================================

Addresses GAP-02: Findings Quality issues identified in research iterations:
- 48% findings are just titles (no actual content)
- 23% are garbage (empty, arXiv labels, truncated)
- 120-char truncation cutting content mid-word

This module provides:
1. Garbage pattern filtering
2. Content extraction from source text (not just titles)
3. Word-boundary-aware truncation (500+ chars default)
4. Quality scoring for findings
5. Deduplication and ranking

Usage:
    from core.research.finding_quality import FindingQualityProcessor

    processor = FindingQualityProcessor()

    # Process raw findings from SDK responses
    clean_findings = processor.process_findings(
        raw_findings=["[exa] Some Article Title", "[exa-h] truncated content..."],
        source_texts=["Full article text here..."],
        highlights=["Key insight from the article..."]
    )

    # Extract findings from source text directly
    findings = processor.extract_findings_from_text(
        text="Long article text with multiple insights...",
        source="exa",
        max_findings=3
    )

Reference:
    GAP-02: https://docs.gap-resolution.io/gap-02-findings-quality
    ADR-027: MCP Configuration Optimization
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FindingQualityConfig:
    """Configuration for finding quality processing.

    Attributes:
        min_length: Minimum character length for valid finding (default: 50)
        max_length: Maximum character length before truncation (default: 500)
        truncation_buffer: Extra chars to scan for word boundary (default: 50)
        min_quality_score: Minimum score to accept finding (default: 0.4)
        max_findings_per_source: Max findings to extract per source (default: 5)
        deduplicate_threshold: Similarity threshold for deduplication (default: 0.8)
    """
    min_length: int = 50
    max_length: int = 500
    truncation_buffer: int = 50
    min_quality_score: float = 0.4
    max_findings_per_source: int = 5
    deduplicate_threshold: float = 0.8


# =============================================================================
# GARBAGE PATTERNS
# =============================================================================

class GarbagePatterns:
    """Patterns for identifying garbage/low-quality findings."""

    # arXiv and academic paper patterns
    ARXIV_PATTERNS = [
        r'^\s*arXiv:\d+\.\d+',
        r'^\s*\[?\d{4}\.\d{4,5}\]?',
        r'^\s*(?:cs|stat|math|physics|q-bio|q-fin)\.\w+',
        r'^\s*submitted to\s+\w+',
        r'^\s*preprint\s+',
    ]

    # Empty/placeholder patterns
    EMPTY_PATTERNS = [
        r'^\s*$',
        r'^[\s\-_\.]+$',
        r'^\s*N/?A\s*$',
        r'^\s*null\s*$',
        r'^\s*undefined\s*$',
        r'^\s*none\s*$',
        r'^\s*\[?\s*\]?\s*$',
    ]

    # URL-only patterns
    URL_ONLY_PATTERNS = [
        r'^\s*https?://\S+\s*$',
        r'^\s*www\.\S+\s*$',
    ]

    # Source tag only patterns (no content after tag)
    TAG_ONLY_PATTERNS = [
        r'^\s*\[[\w\-]+\]\s*$',
        r'^\s*\[[\w\-]+\]\s*\.{3}\s*$',
    ]

    # Truncation artifacts
    TRUNCATION_ARTIFACTS = [
        r'\.\.\.\s*$',
        r'\s+\.\.\.$',
        r'\s+…$',
        r'\s+\w$',  # Single char at end (mid-word cut)
    ]

    # Citation-only patterns
    CITATION_ONLY = [
        r'^\s*\[\d+\]\s*$',
        r'^\s*Citation\s*\d*\s*$',
        r'^\s*Source\s*\d*\s*$',
        r'^\s*Reference\s*\d*\s*$',
    ]

    @classmethod
    def get_all_garbage_patterns(cls) -> List[re.Pattern]:
        """Get all garbage patterns compiled."""
        all_patterns = (
            cls.ARXIV_PATTERNS +
            cls.EMPTY_PATTERNS +
            cls.URL_ONLY_PATTERNS +
            cls.TAG_ONLY_PATTERNS +
            cls.CITATION_ONLY
        )
        return [re.compile(p, re.IGNORECASE) for p in all_patterns]

    @classmethod
    def get_truncation_patterns(cls) -> List[re.Pattern]:
        """Get truncation artifact patterns."""
        return [re.compile(p) for p in cls.TRUNCATION_ARTIFACTS]


# =============================================================================
# TITLE DETECTION
# =============================================================================

class TitleDetector:
    """Detects if a finding is just a title (not actual content)."""

    # Patterns that indicate title-only content
    TITLE_INDICATORS = [
        # Very short (titles typically < 100 chars)
        lambda text: len(text.strip()) < 80,
        # No sentence structure (no period followed by space and capital)
        lambda text: not re.search(r'\.\s+[A-Z]', text),
        # Title case throughout
        lambda text: text.strip() == text.strip().title(),
        # Ends with colon (typical for titles)
        lambda text: text.strip().endswith(':'),
        # Contains only one "sentence" with no verb indicators
        lambda text: len(re.findall(r'[.!?]', text)) <= 1 and not re.search(r'\b(is|are|was|were|has|have|does|do|can|will|should|would|could|may|might)\b', text.lower()),
    ]

    # Keywords that often appear in titles
    TITLE_KEYWORDS = [
        'guide', 'tutorial', 'introduction', 'overview', 'review',
        'comparison', 'analysis', 'study', 'research', 'paper',
        'article', 'post', 'blog', 'news', 'announcement',
    ]

    @classmethod
    def is_likely_title(cls, text: str) -> Tuple[bool, float]:
        """Check if text is likely just a title.

        Returns:
            Tuple of (is_title, confidence)
        """
        text = cls._strip_source_tag(text)

        if not text:
            return True, 1.0

        title_score = 0.0
        checks_passed = 0

        for indicator in cls.TITLE_INDICATORS:
            try:
                if indicator(text):
                    title_score += 0.2
                    checks_passed += 1
            except Exception:
                pass

        # Check for title keywords
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in cls.TITLE_KEYWORDS if kw in text_lower)
        if keyword_matches > 0:
            title_score += min(0.2, keyword_matches * 0.05)

        # Normalize score
        title_score = min(1.0, title_score)

        is_title = title_score > 0.5
        return is_title, title_score

    @staticmethod
    def _strip_source_tag(text: str) -> str:
        """Remove source tags like [exa], [tavily], etc."""
        return re.sub(r'^\s*\[[\w\-]+\]\s*', '', text).strip()


# =============================================================================
# CONTENT EXTRACTION
# =============================================================================

class ContentExtractor:
    """Extracts meaningful findings from source text."""

    # Sentence ending patterns
    SENTENCE_END = re.compile(r'[.!?](?:\s+|$)')

    # Patterns for key insight sentences
    INSIGHT_INDICATORS = [
        r'\b(?:key|main|important|significant|notable|critical)\s+(?:finding|insight|point|takeaway)',
        r'\b(?:research|study|analysis)\s+(?:shows|demonstrates|reveals|indicates|suggests)',
        r'\b(?:results|findings|data)\s+(?:show|indicate|suggest|demonstrate)',
        r'\b(?:we\s+)?(?:found|discovered|observed|noted)\s+that',
        r'\b(?:this|these)\s+(?:shows?|demonstrates?|indicates?|suggests?)',
        r'\b(?:importantly|significantly|notably|interestingly)',
        r'\b(?:in\s+conclusion|to\s+summarize|in\s+summary)',
        r'\b(?:best\s+practice|recommended|optimal)\b',
        r'\b(?:advantage|benefit|improvement|performance)\b',
    ]

    def __init__(self, config: Optional[FindingQualityConfig] = None):
        self.config = config or FindingQualityConfig()
        self._insight_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INSIGHT_INDICATORS
        ]

    def extract_findings(
        self,
        text: str,
        source: str = "unknown",
        max_findings: int = 3
    ) -> List[str]:
        """Extract meaningful findings from source text.

        Args:
            text: Full source text to extract from
            source: Source identifier (exa, tavily, etc.)
            max_findings: Maximum findings to extract

        Returns:
            List of extracted findings with source tags
        """
        if not text or len(text.strip()) < self.config.min_length:
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # Score sentences for insight quality
        scored_sentences = []
        for sent in sentences:
            score = self._score_sentence(sent)
            if score > 0.3 and len(sent) >= 30:
                scored_sentences.append((sent, score))

        # Sort by score and take top findings
        scored_sentences.sort(key=lambda x: -x[1])

        findings = []
        for sent, score in scored_sentences[:max_findings]:
            # Truncate properly if needed
            truncated = self._smart_truncate(sent)
            finding = f"[{source}] {truncated}"
            findings.append(finding)

        return findings

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Basic sentence splitting
        sentences = self.SENTENCE_END.split(text)

        # Clean and filter
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 20:
                # Re-add period if missing
                if not sent[-1] in '.!?':
                    sent += '.'
                result.append(sent)

        return result

    def _score_sentence(self, sentence: str) -> float:
        """Score a sentence for insight quality."""
        score = 0.5  # Base score

        # Check for insight indicators
        for pattern in self._insight_patterns:
            if pattern.search(sentence):
                score += 0.15

        # Penalize very short sentences
        if len(sentence) < 50:
            score -= 0.2

        # Bonus for medium-length sentences (ideal for findings)
        if 100 <= len(sentence) <= 300:
            score += 0.1

        # Penalize sentences that are too long
        if len(sentence) > 500:
            score -= 0.1

        # Check for specific content markers
        lower = sentence.lower()

        # Technical terms bonus
        tech_terms = ['api', 'model', 'algorithm', 'implementation', 'architecture',
                      'performance', 'latency', 'throughput', 'accuracy', 'precision']
        tech_matches = sum(1 for t in tech_terms if t in lower)
        score += min(0.2, tech_matches * 0.05)

        # Numbers/metrics bonus (indicates concrete findings)
        if re.search(r'\d+(?:\.\d+)?%|\d+(?:ms|s|mb|gb|k|m)\b', lower):
            score += 0.15

        return min(1.0, max(0.0, score))

    def _smart_truncate(self, text: str) -> str:
        """Truncate text at word boundary."""
        if len(text) <= self.config.max_length:
            return text

        # Find word boundary near max_length
        truncate_point = self.config.max_length

        # Look back for space
        while truncate_point > self.config.max_length - self.config.truncation_buffer:
            if text[truncate_point] == ' ':
                break
            truncate_point -= 1

        # If no space found, just truncate at max
        if truncate_point <= self.config.max_length - self.config.truncation_buffer:
            truncate_point = self.config.max_length

        truncated = text[:truncate_point].rstrip()

        # Add ellipsis if truncated
        if len(truncated) < len(text):
            truncated += '...'

        return truncated


# =============================================================================
# QUALITY SCORER
# =============================================================================

@dataclass
class QualityScore:
    """Quality assessment for a finding."""
    total_score: float
    is_garbage: bool
    is_title_only: bool
    has_truncation_issues: bool
    content_length: int
    details: Dict[str, Any] = field(default_factory=dict)


class QualityScorer:
    """Scores findings for quality."""

    def __init__(self, config: Optional[FindingQualityConfig] = None):
        self.config = config or FindingQualityConfig()
        self._garbage_patterns = GarbagePatterns.get_all_garbage_patterns()
        self._truncation_patterns = GarbagePatterns.get_truncation_patterns()

    def score(self, finding: str) -> QualityScore:
        """Score a finding for quality.

        Args:
            finding: The finding text to score

        Returns:
            QualityScore with assessment details
        """
        # Strip source tag for analysis
        clean_text = re.sub(r'^\s*\[[\w\-]+\]\s*', '', finding).strip()

        # Check for garbage
        is_garbage = self._is_garbage(clean_text)

        # Check for title-only
        is_title, title_confidence = TitleDetector.is_likely_title(clean_text)

        # Check for truncation issues
        has_truncation = self._has_truncation_issues(clean_text)

        # Calculate score
        score = 1.0

        if is_garbage:
            score = 0.0
        else:
            # Penalize title-only
            if is_title:
                score -= title_confidence * 0.5

            # Penalize truncation
            if has_truncation:
                score -= 0.2

            # Penalize short content
            if len(clean_text) < self.config.min_length:
                score -= 0.3

            # Bonus for good length
            if self.config.min_length <= len(clean_text) <= self.config.max_length:
                score += 0.1

        score = max(0.0, min(1.0, score))

        return QualityScore(
            total_score=score,
            is_garbage=is_garbage,
            is_title_only=is_title,
            has_truncation_issues=has_truncation,
            content_length=len(clean_text),
            details={
                'title_confidence': title_confidence if is_title else 0.0,
                'original_finding': finding[:100] + '...' if len(finding) > 100 else finding,
            }
        )

    def _is_garbage(self, text: str) -> bool:
        """Check if text matches garbage patterns."""
        for pattern in self._garbage_patterns:
            if pattern.match(text):
                return True
        return False

    def _has_truncation_issues(self, text: str) -> bool:
        """Check for truncation artifacts."""
        for pattern in self._truncation_patterns:
            if pattern.search(text):
                return True
        return False


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class FindingQualityProcessor:
    """Main processor for finding quality improvement.

    Combines garbage filtering, content extraction, and quality scoring
    to produce high-quality findings from raw SDK responses.

    Example:
        >>> processor = FindingQualityProcessor()
        >>>
        >>> # From raw findings
        >>> findings = processor.process_findings(
        ...     raw_findings=["[exa] Article Title", "[tavily] Answer text..."],
        ...     source_texts=["Full article content..."],
        ...     highlights=["Key highlight..."]
        ... )
        >>>
        >>> # Direct extraction
        >>> findings = processor.extract_from_source(
        ...     text="Full source text...",
        ...     source="exa"
        ... )
    """

    def __init__(self, config: Optional[FindingQualityConfig] = None):
        self.config = config or FindingQualityConfig()
        self.scorer = QualityScorer(self.config)
        self.extractor = ContentExtractor(self.config)

    def process_findings(
        self,
        raw_findings: List[str],
        source_texts: Optional[List[str]] = None,
        highlights: Optional[List[str]] = None,
        source: str = "unknown"
    ) -> List[str]:
        """Process raw findings to improve quality.

        Args:
            raw_findings: Raw findings from SDK
            source_texts: Optional full source texts for extraction
            highlights: Optional highlights for supplementation
            source: Source identifier

        Returns:
            List of high-quality findings
        """
        quality_findings = []
        seen_content: Set[str] = set()

        # First, score and filter raw findings
        for finding in raw_findings:
            score = self.scorer.score(finding)

            if score.total_score >= self.config.min_quality_score:
                # Clean and potentially re-truncate
                clean = self._clean_finding(finding)
                if clean and clean not in seen_content:
                    quality_findings.append(clean)
                    seen_content.add(clean)

        # If we have source texts and need more findings, extract from them
        if source_texts and len(quality_findings) < self.config.max_findings_per_source:
            needed = self.config.max_findings_per_source - len(quality_findings)
            for text in source_texts:
                if len(quality_findings) >= self.config.max_findings_per_source:
                    break

                extracted = self.extractor.extract_findings(
                    text,
                    source=source,
                    max_findings=needed
                )

                for finding in extracted:
                    clean = self._normalize_for_dedup(finding)
                    if clean not in seen_content:
                        quality_findings.append(finding)
                        seen_content.add(clean)

                        if len(quality_findings) >= self.config.max_findings_per_source:
                            break

        # If we have highlights and still need more, use them
        if highlights and len(quality_findings) < self.config.max_findings_per_source:
            for highlight in highlights:
                if len(quality_findings) >= self.config.max_findings_per_source:
                    break

                # Properly truncate highlight
                truncated = self.extractor._smart_truncate(highlight)
                finding = f"[{source}-highlight] {truncated}"

                score = self.scorer.score(finding)
                if score.total_score >= self.config.min_quality_score:
                    clean = self._normalize_for_dedup(finding)
                    if clean not in seen_content:
                        quality_findings.append(finding)
                        seen_content.add(clean)

        return quality_findings

    def extract_from_source(
        self,
        text: str,
        source: str = "unknown",
        max_findings: Optional[int] = None
    ) -> List[str]:
        """Extract findings directly from source text.

        Args:
            text: Full source text
            source: Source identifier
            max_findings: Maximum findings to extract

        Returns:
            List of extracted findings
        """
        max_f = max_findings or self.config.max_findings_per_source
        return self.extractor.extract_findings(text, source, max_f)

    def filter_garbage(self, findings: List[str]) -> List[str]:
        """Filter garbage findings from a list.

        Args:
            findings: List of findings to filter

        Returns:
            Filtered list with garbage removed
        """
        return [
            f for f in findings
            if not self.scorer.score(f).is_garbage
        ]

    def smart_truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text at word boundary.

        Args:
            text: Text to truncate
            max_length: Maximum length (default: config.max_length)

        Returns:
            Truncated text with proper word boundaries
        """
        if max_length:
            original_max = self.config.max_length
            self.config.max_length = max_length
            result = self.extractor._smart_truncate(text)
            self.config.max_length = original_max
            return result
        return self.extractor._smart_truncate(text)

    def score_finding(self, finding: str) -> QualityScore:
        """Get quality score for a finding.

        Args:
            finding: Finding text to score

        Returns:
            QualityScore with assessment
        """
        return self.scorer.score(finding)

    def _clean_finding(self, finding: str) -> str:
        """Clean a finding (fix truncation, normalize whitespace)."""
        # Normalize whitespace
        finding = ' '.join(finding.split())

        # If it has truncation issues, try to fix
        if self.scorer._has_truncation_issues(finding):
            # Remove trailing truncation artifacts
            finding = re.sub(r'\s+\.\.\.$', '...', finding)
            finding = re.sub(r'\s+…$', '...', finding)
            finding = re.sub(r'\s+\w$', '...', finding)

        return finding

    def _normalize_for_dedup(self, text: str) -> str:
        """Normalize text for deduplication comparison."""
        # Remove source tags
        text = re.sub(r'^\s*\[[\w\-]+\]\s*', '', text)
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_exa_results(
    results: List[Any],
    max_findings: int = 5
) -> List[str]:
    """Process Exa search results to extract quality findings.

    Args:
        results: Exa search results (list of result objects)
        max_findings: Maximum findings to extract

    Returns:
        List of quality findings
    """
    processor = FindingQualityProcessor()

    raw_findings = []
    source_texts = []
    highlights = []

    for r in results:
        # Extract title (but don't use as primary finding)
        if hasattr(r, 'title') and r.title:
            raw_findings.append(f"[exa] {r.title}")

        # Get full text for extraction
        if hasattr(r, 'text') and r.text:
            source_texts.append(r.text)

        # Get highlights
        if hasattr(r, 'highlights') and r.highlights:
            highlights.extend(r.highlights)

    return processor.process_findings(
        raw_findings=raw_findings,
        source_texts=source_texts,
        highlights=highlights,
        source="exa"
    )[:max_findings]


def process_tavily_results(
    data: Dict[str, Any],
    max_findings: int = 3
) -> List[str]:
    """Process Tavily search results to extract quality findings.

    Args:
        data: Tavily API response dict
        max_findings: Maximum findings to extract

    Returns:
        List of quality findings
    """
    processor = FindingQualityProcessor()

    raw_findings = []
    source_texts = []

    # Get answer if present (Tavily's synthesized answer is high quality)
    if data.get('answer'):
        answer = data['answer']
        # Smart truncate the answer
        truncated = processor.smart_truncate(answer, 500)
        raw_findings.append(f"[tavily] {truncated}")

    # Get content from results
    for result in data.get('results', []):
        if result.get('content'):
            source_texts.append(result['content'])

    return processor.process_findings(
        raw_findings=raw_findings,
        source_texts=source_texts,
        source="tavily"
    )[:max_findings]


def process_perplexity_results(
    content: str,
    citations: Optional[List[str]] = None,
    max_findings: int = 3
) -> List[str]:
    """Process Perplexity response to extract quality findings.

    Args:
        content: Response content from Perplexity
        citations: Optional list of citation URLs
        max_findings: Maximum findings to extract

    Returns:
        List of quality findings
    """
    processor = FindingQualityProcessor()

    if not content:
        return []

    # Extract findings from the content directly
    findings = processor.extract_from_source(
        content,
        source="perplexity",
        max_findings=max_findings
    )

    return findings


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main processor
    'FindingQualityProcessor',
    # Configuration
    'FindingQualityConfig',
    # Scoring
    'QualityScorer',
    'QualityScore',
    # Extraction
    'ContentExtractor',
    # Detection
    'TitleDetector',
    'GarbagePatterns',
    # Convenience functions
    'process_exa_results',
    'process_tavily_results',
    'process_perplexity_results',
]
