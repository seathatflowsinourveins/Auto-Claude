"""
Content Detection Module - V36 Consolidated API

This module provides unified content detection functionality for the UNLEASH platform.
It consolidates the various detect_content_type implementations into a clean API.

Architecture Decision: ADR-003 - Unified Content Detection
Canonical Source: platform/core/advanced_memory.py

Usage:
    from core.content_detection import detect_content_type, ContentType

    content_type = detect_content_type("def hello(): print('world')")
    # Returns: ContentType.CODE

Advanced Usage (with confidence scoring):
    from core.content_detection import detect_with_confidence

    result = detect_with_confidence("def hello(): print('world')")
    # Returns: ContentConfidence(content_type=CODE, confidence=0.95, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# Try to import from canonical source (advanced_memory.py)
try:
    from ..advanced_memory import (
        ContentType,
        detect_content_type as _detect_v124,
        detect_content_type_v126 as _detect_v126,
        detect_content_type_hybrid as _detect_hybrid,
        ContentConfidence,
    )
    _ADVANCED_MEMORY_AVAILABLE = True
except ImportError:
    _ADVANCED_MEMORY_AVAILABLE = False

    # Fallback definitions
    class ContentType(Enum):
        """Content type classification."""
        CODE = "code"
        TEXT = "text"
        MULTILINGUAL = "multilingual"
        MIXED = "mixed"
        UNKNOWN = "unknown"

    @dataclass
    class ContentConfidence:
        """Content detection result with confidence scores."""
        content_type: ContentType
        confidence: float
        code_score: float
        multilingual_score: float
        text_score: float


# Try to import Chonkie-based detection from benchmarking
try:
    from ..benchmarking import (
        detect_content_type_v129 as _detect_v129,
        ChonkieAnalysis,
        is_chonkie_available,
    )
    _CHONKIE_AVAILABLE = is_chonkie_available()
except ImportError:
    _CHONKIE_AVAILABLE = False
    ChonkieAnalysis = None


def detect_content_type(text: str, version: str = "auto") -> ContentType:
    """
    Detect the type of content for optimal model/embedding selection.

    Args:
        text: Input text to analyze
        version: Detection algorithm version:
            - "auto" (default): Best available algorithm
            - "v124": Basic detection (fast, low memory)
            - "v126": Confidence-based (returns ContentType only)
            - "hybrid": Multi-signal fusion
            - "v129": Chonkie-based (if available)

    Returns:
        ContentType indicating the dominant content type

    Example:
        >>> detect_content_type("def hello(): pass")
        ContentType.CODE

        >>> detect_content_type("Hello, world!")
        ContentType.TEXT
    """
    if not _ADVANCED_MEMORY_AVAILABLE:
        # Minimal fallback detection
        return _minimal_detect(text)

    if version == "auto":
        # Use best available: v126 for production (confidence-based)
        result = _detect_v126(text)
        return result.content_type
    elif version == "v124":
        return _detect_v124(text)
    elif version == "v126":
        result = _detect_v126(text)
        return result.content_type
    elif version == "hybrid":
        return _detect_hybrid(text)
    elif version == "v129" and _CHONKIE_AVAILABLE:
        result = _detect_v129(text)
        return result.content_type
    else:
        # Default to v126
        result = _detect_v126(text)
        return result.content_type


def detect_with_confidence(text: str, version: str = "v126") -> ContentConfidence:
    """
    Detect content type with confidence scores.

    Args:
        text: Input text to analyze
        version: Detection algorithm version ("v126" or "auto")

    Returns:
        ContentConfidence with type and per-category scores

    Example:
        >>> result = detect_with_confidence("def hello(): pass")
        >>> result.content_type
        ContentType.CODE
        >>> result.confidence
        0.95
    """
    if not _ADVANCED_MEMORY_AVAILABLE:
        # Minimal fallback
        content_type = _minimal_detect(text)
        return ContentConfidence(
            content_type=content_type,
            confidence=0.5,
            code_score=0.5 if content_type == ContentType.CODE else 0.0,
            multilingual_score=0.0,
            text_score=0.5 if content_type == ContentType.TEXT else 0.0,
        )

    return _detect_v126(text)


def _minimal_detect(text: str) -> ContentType:
    """Minimal fallback content detection when advanced_memory is unavailable."""
    if not text or len(text.strip()) < 5:
        return ContentType.UNKNOWN

    # Simple code detection
    code_indicators = [
        "def ", "class ", "function ", "import ", "from ", "const ",
        "let ", "var ", "fn ", "pub ", "struct ", "enum ",
        "if (", "for (", "while (", "return ", "async ",
    ]

    text_lower = text.lower()
    code_count = sum(1 for indicator in code_indicators if indicator in text_lower)

    if code_count >= 2:
        return ContentType.CODE
    elif code_count >= 1 and ('```' in text or '    ' in text or '\t' in text):
        return ContentType.CODE
    else:
        return ContentType.TEXT


# Re-export for convenience
__all__ = [
    "ContentType",
    "ContentConfidence",
    "detect_content_type",
    "detect_with_confidence",
]
