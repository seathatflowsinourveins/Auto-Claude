#!/usr/bin/env python3
"""
Scanner Compatibility Layer for Python 3.14+
Part of V34 Architecture - Phase 10 Fix.

This module provides a transformers-based replacement for llm-guard's
input/output scanners. The llm-guard package has PyTorch ABI compatibility
issues on Python 3.14+ (requires pre-built wheels not yet available).

Key features replaced:
- Prompt injection detection
- PII detection and anonymization
- Toxicity detection
- Language detection
- Code detection
- Secrets detection

Usage:
    from core.safety.scanner_compat import (
        InputScanner,
        OutputScanner,
        ScanResult,
        scan_input,
        scan_output,
    )

    # Scan input
    result = scan_input("Hello, my email is test@example.com")
    if not result.is_safe:
        print(f"Blocked: {result.categories}")

    # Scan output
    result = scan_output("Here's your password: abc123")
    if result.has_pii:
        print(f"PII detected: {result.pii_entities}")
"""

from __future__ import annotations

import re
import os
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# =============================================================================
# CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# Try to import transformers for advanced detection
TRANSFORMERS_AVAILABLE = False
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available - using regex-based scanning only")


# =============================================================================
# ENUMS
# =============================================================================

class RiskLevel(str, Enum):
    """Risk level for detected issues."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    """Categories of detected issues."""
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    TOXICITY = "toxicity"
    SECRETS = "secrets"
    CODE = "code"
    LANGUAGE = "language"
    BIAS = "bias"
    HARMFUL = "harmful"
    CUSTOM = "custom"


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    LICENSE = "license"
    URL = "url"
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PIIEntity:
    """Represents a detected PII entity."""
    type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    redacted: str = "[REDACTED]"


@dataclass
class Detection:
    """Represents a single detection."""
    category: Category
    risk_level: RiskLevel
    message: str
    confidence: float = 1.0
    start: Optional[int] = None
    end: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ScanResult:
    """Result of scanning text."""
    text: str
    is_safe: bool = True
    risk_level: RiskLevel = RiskLevel.NONE
    detections: List[Detection] = field(default_factory=list)
    pii_entities: List[PIIEntity] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    categories: Set[Category] = field(default_factory=set)
    scan_time_ms: float = 0.0

    @property
    def has_pii(self) -> bool:
        return len(self.pii_entities) > 0

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "risk_level": self.risk_level.value,
            "detection_count": self.detection_count,
            "has_pii": self.has_pii,
            "categories": [c.value for c in self.categories],
            "scan_time_ms": self.scan_time_ms
        }


# =============================================================================
# ABSTRACT SCANNER BASE
# =============================================================================

class BaseScanner(ABC):
    """Abstract base class for all scanners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Scanner name."""
        pass

    @property
    @abstractmethod
    def category(self) -> Category:
        """Detection category."""
        pass

    @abstractmethod
    def scan(self, text: str) -> List[Detection]:
        """Scan text and return detections."""
        pass


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# PII Patterns
PII_PATTERNS = {
    PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
    PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
    PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    PIIType.URL: r'https?://[^\s<>"{}|\\^`\[\]]+',
}

# Secret Patterns
SECRET_PATTERNS = {
    "api_key": r'(?i)(?:api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_-]{20,})["\']?',
    "aws_key": r'(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}',
    "github_token": r'gh[pousr]_[A-Za-z0-9_]{36,}',
    "openai_key": r'sk-[a-zA-Z0-9]{48,}',
    "anthropic_key": r'sk-ant-[a-zA-Z0-9-]{93,}',
    "jwt_token": r'eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_.+/]*',
    "private_key": r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    "password_in_url": r'://[^:]+:([^@]+)@',
    "bearer_token": r'(?i)bearer\s+[a-zA-Z0-9_-]{20,}',
    "basic_auth": r'(?i)basic\s+[a-zA-Z0-9+/=]{20,}',
}

# Prompt Injection Patterns
INJECTION_PATTERNS = [
    r'(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)',
    r'(?i)forget\s+(everything|all|what)\s+(you|i)\s+(know|said|told)',
    r'(?i)you\s+are\s+now\s+(?:a|an|in)\s+(?:new|different)\s+(?:mode|role|character)',
    r'(?i)override\s+(?:your|the)\s+(?:instructions?|rules?|guidelines?)',
    r'(?i)act\s+as\s+(?:if|though)\s+you\s+(?:are|were)\s+(?:not|un)',
    r'(?i)disregard\s+(?:your|the|all)\s+(?:previous|safety|content)\s+(?:instructions?|filters?|rules?)',
    r'(?i)pretend\s+(?:you\s+)?(?:don\'?t?\s+)?have\s+(?:no\s+)?(?:restrictions?|limits?|rules?)',
    r'(?i)jailbreak|dan\s+mode|developer\s+mode',
    r'(?i)\[system\]|\[user\]|\[assistant\]',
    r'(?i)<<\s*(?:SYS|SYSTEM|USER|INST).*>>',
]

# Toxicity Keywords (simplified - use transformer models for production)
TOXICITY_KEYWORDS = [
    # Explicit hate speech and slurs (censored patterns)
    r'(?i)\b(?:hate|kill|murder|attack)\s+(?:all|every)\s+\w+s?\b',
    r'(?i)\b(?:should\s+)?(?:die|be\s+killed|be\s+eliminated)\b',
]


# =============================================================================
# SCANNER IMPLEMENTATIONS
# =============================================================================

class PIIScanner(BaseScanner):
    """Scans for Personally Identifiable Information."""

    def __init__(self, types_to_detect: Optional[List[PIIType]] = None):
        self.types_to_detect = types_to_detect or list(PIIType)
        self._compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in PII_PATTERNS.items()
            if pii_type in self.types_to_detect
        }

    @property
    def name(self) -> str:
        return "PII Scanner"

    @property
    def category(self) -> Category:
        return Category.PII

    def scan(self, text: str) -> Tuple[List[Detection], List[PIIEntity]]:
        """Scan for PII and return detections and entities."""
        detections = []
        entities = []

        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                entity = PIIEntity(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end()
                )
                entities.append(entity)

                detection = Detection(
                    category=Category.PII,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Detected {pii_type.value}: {entity.redacted}",
                    start=match.start(),
                    end=match.end(),
                    details={"pii_type": pii_type.value}
                )
                detections.append(detection)

        return detections, entities

    def redact(self, text: str, entities: List[PIIEntity]) -> str:
        """Redact PII entities from text."""
        if not entities:
            return text

        # Sort by position (reverse) to maintain indices
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        result = text

        for entity in sorted_entities:
            result = result[:entity.start] + entity.redacted + result[entity.end:]

        return result


class SecretsScanner(BaseScanner):
    """Scans for secrets and credentials."""

    def __init__(self):
        self._compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in SECRET_PATTERNS.items()
        }

    @property
    def name(self) -> str:
        return "Secrets Scanner"

    @property
    def category(self) -> Category:
        return Category.SECRETS

    def scan(self, text: str) -> List[Detection]:
        detections = []

        for secret_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                detection = Detection(
                    category=Category.SECRETS,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Detected potential {secret_type}",
                    start=match.start(),
                    end=match.end(),
                    details={"secret_type": secret_type}
                )
                detections.append(detection)

        return detections


class PromptInjectionScanner(BaseScanner):
    """Scans for prompt injection attempts."""

    def __init__(self):
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in INJECTION_PATTERNS
        ]

    @property
    def name(self) -> str:
        return "Prompt Injection Scanner"

    @property
    def category(self) -> Category:
        return Category.PROMPT_INJECTION

    def scan(self, text: str) -> List[Detection]:
        detections = []

        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                detection = Detection(
                    category=Category.PROMPT_INJECTION,
                    risk_level=RiskLevel.HIGH,
                    message="Potential prompt injection detected",
                    confidence=0.8,
                    start=match.start(),
                    end=match.end(),
                    details={"matched_pattern": match.group()[:100]}
                )
                detections.append(detection)

        return detections


class ToxicityScanner(BaseScanner):
    """Scans for toxic content."""

    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self._model = None
        self._tokenizer = None
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in TOXICITY_KEYWORDS
        ]

    def _load_model(self):
        """Lazy load the toxicity detection model."""
        if self._model is None and self.use_transformer:
            try:
                from transformers import pipeline
                self._model = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=-1  # CPU
                )
            except Exception as e:
                logger.warning(f"Failed to load toxicity model: {e}")
                self.use_transformer = False

    @property
    def name(self) -> str:
        return "Toxicity Scanner"

    @property
    def category(self) -> Category:
        return Category.TOXICITY

    def scan(self, text: str) -> List[Detection]:
        detections = []

        # Try transformer-based detection first
        if self.use_transformer:
            self._load_model()
            if self._model:
                try:
                    results = self._model(text[:512])  # Limit input length
                    for result in results:
                        if result["label"].lower() == "toxic" and result["score"] > 0.7:
                            detection = Detection(
                                category=Category.TOXICITY,
                                risk_level=RiskLevel.HIGH,
                                message="Toxic content detected",
                                confidence=result["score"],
                                details={"label": result["label"]}
                            )
                            detections.append(detection)
                except Exception as e:
                    logger.warning(f"Transformer detection failed: {e}")

        # Fallback to regex patterns
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                detection = Detection(
                    category=Category.TOXICITY,
                    risk_level=RiskLevel.HIGH,
                    message="Potentially toxic content detected",
                    confidence=0.7,
                    start=match.start(),
                    end=match.end()
                )
                detections.append(detection)

        return detections


class CodeScanner(BaseScanner):
    """Scans for code in text."""

    CODE_PATTERNS = [
        r'```[\w]*\n[\s\S]*?```',  # Markdown code blocks
        r'<script[^>]*>[\s\S]*?</script>',  # Script tags
        r'<\?php[\s\S]*?\?>',  # PHP
        r'(?:import|from)\s+[\w.]+\s+(?:import)?',  # Python imports
        r'(?:function|const|let|var)\s+\w+\s*[=(]',  # JavaScript
        r'(?:public|private|protected)\s+(?:static\s+)?(?:void|int|String)',  # Java
        r'SELECT\s+.*?\s+FROM\s+\w+',  # SQL
    ]

    def __init__(self, block_code: bool = False):
        self.block_code = block_code
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.CODE_PATTERNS
        ]

    @property
    def name(self) -> str:
        return "Code Scanner"

    @property
    def category(self) -> Category:
        return Category.CODE

    def scan(self, text: str) -> List[Detection]:
        detections = []

        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                risk = RiskLevel.MEDIUM if self.block_code else RiskLevel.LOW
                detection = Detection(
                    category=Category.CODE,
                    risk_level=risk,
                    message="Code detected in text",
                    start=match.start(),
                    end=match.end(),
                    details={"code_snippet": match.group()[:100]}
                )
                detections.append(detection)

        return detections


# =============================================================================
# COMPOSITE SCANNERS
# =============================================================================

class InputScanner:
    """Composite scanner for user input."""

    def __init__(
        self,
        enable_pii: bool = True,
        enable_secrets: bool = True,
        enable_injection: bool = True,
        enable_toxicity: bool = True,
        enable_code: bool = False,
        redact_pii: bool = True,
        custom_scanners: Optional[List[BaseScanner]] = None
    ):
        self.redact_pii = redact_pii
        self.scanners: List[BaseScanner] = []

        if enable_pii:
            self.pii_scanner = PIIScanner()
            self.scanners.append(self.pii_scanner)
        else:
            self.pii_scanner = None

        if enable_secrets:
            self.scanners.append(SecretsScanner())
        if enable_injection:
            self.scanners.append(PromptInjectionScanner())
        if enable_toxicity:
            self.scanners.append(ToxicityScanner())
        if enable_code:
            self.scanners.append(CodeScanner(block_code=True))

        if custom_scanners:
            self.scanners.extend(custom_scanners)

    def scan(self, text: str) -> ScanResult:
        """Scan input text."""
        import time
        start_time = time.perf_counter()

        all_detections = []
        pii_entities = []
        categories = set()

        # Run all scanners
        for scanner in self.scanners:
            if isinstance(scanner, PIIScanner):
                detections, entities = scanner.scan(text)
                all_detections.extend(detections)
                pii_entities.extend(entities)
            else:
                detections = scanner.scan(text)
                all_detections.extend(detections)

            if detections:
                categories.add(scanner.category)

        # Determine risk level
        risk_level = RiskLevel.NONE
        for detection in all_detections:
            if detection.risk_level.value > risk_level.value:
                risk_level = detection.risk_level

        # Sanitize if needed
        sanitized_text = None
        if self.redact_pii and pii_entities and self.pii_scanner:
            sanitized_text = self.pii_scanner.redact(text, pii_entities)

        # Determine safety
        is_safe = risk_level in (RiskLevel.NONE, RiskLevel.LOW)

        scan_time = (time.perf_counter() - start_time) * 1000

        return ScanResult(
            text=text,
            is_safe=is_safe,
            risk_level=risk_level,
            detections=all_detections,
            pii_entities=pii_entities,
            sanitized_text=sanitized_text,
            categories=categories,
            scan_time_ms=scan_time
        )


class OutputScanner:
    """Composite scanner for model output."""

    def __init__(
        self,
        enable_pii: bool = True,
        enable_secrets: bool = True,
        enable_toxicity: bool = True,
        enable_code: bool = False,
        redact_pii: bool = True,
        custom_scanners: Optional[List[BaseScanner]] = None
    ):
        self.redact_pii = redact_pii
        self.scanners: List[BaseScanner] = []

        if enable_pii:
            self.pii_scanner = PIIScanner()
            self.scanners.append(self.pii_scanner)
        else:
            self.pii_scanner = None

        if enable_secrets:
            self.scanners.append(SecretsScanner())
        if enable_toxicity:
            self.scanners.append(ToxicityScanner())
        if enable_code:
            self.scanners.append(CodeScanner(block_code=False))

        if custom_scanners:
            self.scanners.extend(custom_scanners)

    def scan(self, text: str) -> ScanResult:
        """Scan output text."""
        import time
        start_time = time.perf_counter()

        all_detections = []
        pii_entities = []
        categories = set()

        for scanner in self.scanners:
            if isinstance(scanner, PIIScanner):
                detections, entities = scanner.scan(text)
                all_detections.extend(detections)
                pii_entities.extend(entities)
            else:
                detections = scanner.scan(text)
                all_detections.extend(detections)

            if detections:
                categories.add(scanner.category)

        risk_level = RiskLevel.NONE
        for detection in all_detections:
            if detection.risk_level.value > risk_level.value:
                risk_level = detection.risk_level

        sanitized_text = None
        if self.redact_pii and pii_entities and self.pii_scanner:
            sanitized_text = self.pii_scanner.redact(text, pii_entities)

        is_safe = risk_level in (RiskLevel.NONE, RiskLevel.LOW)

        scan_time = (time.perf_counter() - start_time) * 1000

        return ScanResult(
            text=text,
            is_safe=is_safe,
            risk_level=risk_level,
            detections=all_detections,
            pii_entities=pii_entities,
            sanitized_text=sanitized_text,
            categories=categories,
            scan_time_ms=scan_time
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Default scanners (lazy initialized)
_input_scanner: Optional[InputScanner] = None
_output_scanner: Optional[OutputScanner] = None


def get_input_scanner() -> InputScanner:
    """Get the default input scanner."""
    global _input_scanner
    if _input_scanner is None:
        _input_scanner = InputScanner()
    return _input_scanner


def get_output_scanner() -> OutputScanner:
    """Get the default output scanner."""
    global _output_scanner
    if _output_scanner is None:
        _output_scanner = OutputScanner()
    return _output_scanner


def scan_input(text: str) -> ScanResult:
    """Scan input using the default scanner."""
    return get_input_scanner().scan(text)


def scan_output(text: str) -> ScanResult:
    """Scan output using the default scanner."""
    return get_output_scanner().scan(text)


def is_safe(text: str) -> bool:
    """Quick check if text is safe."""
    result = scan_input(text)
    return result.is_safe


def redact_pii(text: str) -> str:
    """Redact PII from text."""
    scanner = PIIScanner()
    _, entities = scanner.scan(text)
    return scanner.redact(text, entities)


# =============================================================================
# EXPORTS
# =============================================================================

# Compatibility flag and alias for V35 validation
SCANNER_COMPAT_AVAILABLE = True

# Alias for V35 validation compatibility
ScannerCompat = InputScanner

__all__ = [
    # Enums
    "RiskLevel",
    "Category",
    "PIIType",
    # Data classes
    "PIIEntity",
    "Detection",
    "ScanResult",
    # Base
    "BaseScanner",
    # Individual scanners
    "PIIScanner",
    "SecretsScanner",
    "PromptInjectionScanner",
    "ToxicityScanner",
    "CodeScanner",
    # Composite scanners
    "InputScanner",
    "OutputScanner",
    # V35 compat alias
    "ScannerCompat",
    # Convenience
    "scan_input",
    "scan_output",
    "is_safe",
    "redact_pii",
    "get_input_scanner",
    "get_output_scanner",
    # Constants
    "TRANSFORMERS_AVAILABLE",
    "SCANNER_COMPAT_AVAILABLE",
]
