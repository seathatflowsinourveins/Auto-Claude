"""
Hallucination Guard: Real-Time Hallucination Detection for RAG Pipelines

This module provides a production-ready hallucination guard that can be integrated
into RAG pipelines for post-generation verification. It uses claims extraction
and context verification to detect hallucinated content in real-time.

Key Features:
- Claims Extraction: Extract factual claims from generated responses
- Context Verification: Verify each claim against retrieved context
- Confidence Scoring: Provide granular confidence scores
- Flagged Claims: Return specific claims that appear hallucinated
- Pipeline Integration: Easy integration with RAGPipeline

Architecture:
    Generated Response + Retrieved Context
                    |
                    v
    +------------------------------------------+
    |          HallucinationGuard              |
    |------------------------------------------|
    | - ClaimsExtractor (extract claims)       |
    | - ClaimVerifier (verify against context) |
    | - ConfidenceScorer (calculate scores)    |
    +------------------------------------------+
                    |
                    v
            HallucinationResult
            (confidence, flagged_claims, verdict)

Integration:
    from core.rag.hallucination_guard import HallucinationGuard, GuardConfig

    guard = HallucinationGuard(llm=judge_llm, config=GuardConfig())

    # Check a response
    result = await guard.detect(
        response="The answer claims X, Y, and Z...",
        context=["Context chunk 1...", "Context chunk 2..."]
    )

    if result.has_hallucination:
        print(f"WARNING: Hallucination detected!")
        print(f"Confidence: {result.confidence:.2f}")
        for claim in result.flagged_claims:
            print(f"  - {claim.text}: {claim.verdict}")
    else:
        print(f"Response verified. Confidence: {result.confidence:.2f}")

Pipeline Integration:
    from core.rag.pipeline import RAGPipeline
    from core.rag.hallucination_guard import HallucinationGuard

    pipeline = RAGPipeline(llm=llm, retrievers=retrievers)
    guard = HallucinationGuard(llm=judge_llm)

    # Add guard to pipeline
    pipeline.set_hallucination_guard(guard)

    # Or use manually after generation
    result = await pipeline.run("query")
    guard_result = await guard.detect(result.response, result.contexts_used)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers used in hallucination detection."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ClaimVerdict(str, Enum):
    """Verdict for a claim verification."""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class HallucinationSeverity(str, Enum):
    """Severity level of hallucination."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GuardConfig:
    """Configuration for the hallucination guard.

    Attributes:
        confidence_threshold: Minimum confidence to pass (default: 0.7)
        max_claims: Maximum claims to extract and verify (default: 15)
        batch_size: Batch size for parallel verification (default: 5)
        strict_mode: Fail on any unsupported claim (default: False)
        partial_support_weight: Weight for partially supported claims (default: 0.5)
        enable_reasoning: Include reasoning in results (default: True)
        temperature: LLM temperature for verification (default: 0.0)
        timeout_seconds: Timeout for verification (default: 30.0)
        min_claim_length: Minimum characters for a valid claim (default: 10)
    """
    confidence_threshold: float = 0.7
    max_claims: int = 15
    batch_size: int = 5
    strict_mode: bool = False
    partial_support_weight: float = 0.5
    enable_reasoning: bool = True
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    min_claim_length: int = 10


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedClaim:
    """A factual claim extracted from a response."""
    text: str
    source_sentence: str
    claim_type: str = "factual"
    index: int = 0


@dataclass
class VerifiedClaim:
    """A claim with verification result."""
    text: str
    source_sentence: str
    verdict: ClaimVerdict
    confidence: float
    supporting_evidence: Optional[str] = None
    reasoning: str = ""
    claim_type: str = "factual"


@dataclass
class HallucinationResult:
    """Result from hallucination detection.

    Attributes:
        has_hallucination: Whether hallucination was detected
        confidence: Overall confidence score (0-1, higher = less hallucination)
        severity: Severity level of any hallucination
        flagged_claims: List of claims that appear hallucinated
        verified_claims: All verified claims with results
        total_claims: Total number of claims extracted
        supported_claims: Number of supported claims
        unsupported_claims: Number of unsupported claims
        reasoning: Overall reasoning for the verdict
        metadata: Additional metadata
    """
    has_hallucination: bool
    confidence: float
    severity: HallucinationSeverity
    flagged_claims: List[VerifiedClaim] = field(default_factory=list)
    verified_claims: List[VerifiedClaim] = field(default_factory=list)
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    partially_supported_claims: int = 0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_hallucination": self.has_hallucination,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "partially_supported_claims": self.partially_supported_claims,
            "flagged_claims": [
                {
                    "text": c.text,
                    "verdict": c.verdict.value,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                }
                for c in self.flagged_claims
            ],
            "reasoning": self.reasoning,
        }


# =============================================================================
# PROMPTS
# =============================================================================

class GuardPrompts:
    """Prompt templates for hallucination detection."""

    EXTRACT_CLAIMS = """Extract all factual claims from the following response.
A claim is a single, verifiable statement of fact that can be checked against the context.

Response:
{response}

Instructions:
1. Break down the response into individual factual claims
2. Each claim should be self-contained and verifiable
3. Exclude opinions, hedged statements ("might", "could"), and meta-commentary
4. Include claims about numbers, dates, names, relationships, and events
5. Number each claim starting from 1

Output format (one claim per line):
CLAIM 1: [first factual claim]
CLAIM 2: [second factual claim]
...

Claims:"""

    VERIFY_CLAIM_BATCH = """Verify whether each of the following claims is supported by the provided context.

Context:
{context}

Claims to verify:
{claims}

For EACH claim, determine:
1. Is it SUPPORTED (explicitly or clearly implied in context)?
2. Is it NOT_SUPPORTED (contradicted or absent from context)?
3. Is it PARTIALLY_SUPPORTED (some aspects supported, others not)?
4. Is the context INSUFFICIENT to verify this claim?

Output format for each claim (one per line):
CLAIM {number}: {verdict} | CONFIDENCE: {0.0-1.0} | EVIDENCE: {quote or "N/A"} | REASONING: {brief explanation}

Where verdict is one of: SUPPORTED, NOT_SUPPORTED, PARTIALLY_SUPPORTED, INSUFFICIENT_CONTEXT

Verifications:"""

    VERIFY_SINGLE_CLAIM = """Verify if this claim is supported by the provided context.

Claim: {claim}

Context:
{context}

Instructions:
1. Check if the claim can be verified from the context
2. Look for explicit statements or clear implications
3. Be strict: if the context doesn't clearly support the claim, mark as NOT_SUPPORTED
4. Consider both direct statements and reasonable inferences

Output:
VERDICT: [SUPPORTED/NOT_SUPPORTED/PARTIALLY_SUPPORTED/INSUFFICIENT_CONTEXT]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [direct quote from context that supports/contradicts, or "N/A"]
REASONING: [brief explanation of your verdict]"""

    OVERALL_ASSESSMENT = """Based on the claim verification results, provide an overall assessment.

Original Response:
{response}

Verification Results:
{verification_summary}

Provide:
1. Overall verdict on hallucination presence
2. Severity assessment (NONE/LOW/MEDIUM/HIGH/CRITICAL)
3. Brief reasoning

Output:
HAS_HALLUCINATION: [YES/NO]
SEVERITY: [NONE/LOW/MEDIUM/HIGH/CRITICAL]
REASONING: [brief overall assessment]"""


# =============================================================================
# CLAIMS EXTRACTOR
# =============================================================================

class ClaimsExtractor:
    """Extracts factual claims from responses."""

    def __init__(
        self,
        llm: LLMProvider,
        max_claims: int = 15,
        min_claim_length: int = 10
    ):
        self.llm = llm
        self.max_claims = max_claims
        self.min_claim_length = min_claim_length

    async def extract(self, response: str) -> List[ExtractedClaim]:
        """Extract factual claims from a response."""
        prompt = GuardPrompts.EXTRACT_CLAIMS.format(response=response)

        try:
            result = await self.llm.generate(
                prompt,
                max_tokens=1000,
                temperature=0.0
            )
            return self._parse_claims(result, response)
        except Exception as e:
            logger.warning(f"Claims extraction failed: {e}")
            return self._fallback_extraction(response)

    def _parse_claims(
        self,
        llm_response: str,
        original_response: str
    ) -> List[ExtractedClaim]:
        """Parse LLM response into ExtractedClaim objects."""
        claims = []
        lines = llm_response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "CLAIM N:" pattern
            match = re.match(r'^CLAIM\s*(\d+)\s*:\s*(.+)$', line, re.IGNORECASE)
            if match:
                claim_idx = int(match.group(1))
                claim_text = match.group(2).strip()

                if claim_text and len(claim_text) >= self.min_claim_length:
                    # Find the source sentence in original response
                    source = self._find_source_sentence(claim_text, original_response)

                    claims.append(ExtractedClaim(
                        text=claim_text,
                        source_sentence=source,
                        claim_type="factual",
                        index=claim_idx
                    ))

                if len(claims) >= self.max_claims:
                    break

        return claims

    def _find_source_sentence(self, claim: str, response: str) -> str:
        """Find the source sentence for a claim in the original response."""
        # Split response into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)

        # Find best matching sentence
        claim_words = set(claim.lower().split())
        best_match = claim
        best_overlap = 0

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = sentence

        return best_match

    def _fallback_extraction(self, response: str) -> List[ExtractedClaim]:
        """Fallback extraction using sentence splitting."""
        sentences = re.split(r'(?<=[.!?])\s+', response)
        claims = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            # Skip short sentences and obvious non-factual content
            if len(sent) >= self.min_claim_length:
                # Skip sentences that are clearly opinions or hedged
                hedges = ['might', 'could', 'perhaps', 'maybe', 'i think', 'in my opinion']
                if not any(h in sent.lower() for h in hedges):
                    claims.append(ExtractedClaim(
                        text=sent,
                        source_sentence=sent,
                        claim_type="sentence",
                        index=i + 1
                    ))

            if len(claims) >= self.max_claims:
                break

        return claims


# =============================================================================
# CLAIM VERIFIER
# =============================================================================

class ClaimVerifier:
    """Verifies claims against context."""

    def __init__(
        self,
        llm: LLMProvider,
        config: GuardConfig
    ):
        self.llm = llm
        self.config = config

    async def verify_single(
        self,
        claim: ExtractedClaim,
        context: List[str]
    ) -> VerifiedClaim:
        """Verify a single claim against context."""
        combined_context = "\n\n".join(context)

        # Truncate context if too long
        max_context_chars = 4000
        if len(combined_context) > max_context_chars:
            combined_context = combined_context[:max_context_chars] + "..."

        prompt = GuardPrompts.VERIFY_SINGLE_CLAIM.format(
            claim=claim.text,
            context=combined_context
        )

        try:
            result = await self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=self.config.temperature
            )
            return self._parse_single_verification(claim, result)
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return VerifiedClaim(
                text=claim.text,
                source_sentence=claim.source_sentence,
                verdict=ClaimVerdict.INSUFFICIENT_CONTEXT,
                confidence=0.5,
                reasoning=f"Verification error: {e}",
                claim_type=claim.claim_type
            )

    def _parse_single_verification(
        self,
        claim: ExtractedClaim,
        response: str
    ) -> VerifiedClaim:
        """Parse single verification response."""
        response_upper = response.upper()

        # Parse verdict
        verdict = ClaimVerdict.INSUFFICIENT_CONTEXT
        if "VERDICT: SUPPORTED" in response_upper or "VERDICT:SUPPORTED" in response_upper:
            verdict = ClaimVerdict.SUPPORTED
        elif "NOT_SUPPORTED" in response_upper or "NOT SUPPORTED" in response_upper:
            verdict = ClaimVerdict.NOT_SUPPORTED
        elif "PARTIALLY_SUPPORTED" in response_upper or "PARTIALLY SUPPORTED" in response_upper:
            verdict = ClaimVerdict.PARTIALLY_SUPPORTED
        elif "INSUFFICIENT" in response_upper:
            verdict = ClaimVerdict.INSUFFICIENT_CONTEXT

        # Parse confidence
        confidence = 0.5
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        # Parse evidence
        evidence = None
        evidence_match = re.search(
            r'EVIDENCE:\s*(.+?)(?:REASONING:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if evidence_match:
            ev = evidence_match.group(1).strip()
            if ev.upper() != "N/A":
                evidence = ev

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return VerifiedClaim(
            text=claim.text,
            source_sentence=claim.source_sentence,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence=evidence,
            reasoning=reasoning,
            claim_type=claim.claim_type
        )

    async def verify_batch(
        self,
        claims: List[ExtractedClaim],
        context: List[str]
    ) -> List[VerifiedClaim]:
        """Verify multiple claims in parallel batches."""
        verified_claims = []

        # Process in batches
        for i in range(0, len(claims), self.config.batch_size):
            batch = claims[i:i + self.config.batch_size]

            # Run batch verification in parallel
            tasks = [self.verify_single(claim, context) for claim in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, VerifiedClaim):
                    verified_claims.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Batch verification failed: {result}")

        return verified_claims


# =============================================================================
# CONFIDENCE SCORER
# =============================================================================

class ConfidenceScorer:
    """Calculates overall confidence scores."""

    def __init__(self, config: GuardConfig):
        self.config = config

    def calculate(
        self,
        verified_claims: List[VerifiedClaim]
    ) -> Tuple[float, HallucinationSeverity]:
        """Calculate overall confidence and severity.

        Returns:
            Tuple of (confidence_score, severity)
        """
        if not verified_claims:
            return 1.0, HallucinationSeverity.NONE

        total = len(verified_claims)
        supported = sum(
            1 for c in verified_claims
            if c.verdict == ClaimVerdict.SUPPORTED
        )
        partial = sum(
            1 for c in verified_claims
            if c.verdict == ClaimVerdict.PARTIALLY_SUPPORTED
        )
        unsupported = sum(
            1 for c in verified_claims
            if c.verdict == ClaimVerdict.NOT_SUPPORTED
        )

        # Calculate weighted confidence
        # Supported = 1.0, Partial = 0.5, Unsupported = 0, Insufficient = 0.5
        weighted_score = (
            supported * 1.0 +
            partial * self.config.partial_support_weight +
            (total - supported - partial - unsupported) * 0.5  # insufficient
        ) / total

        confidence = weighted_score

        # Determine severity
        unsupported_ratio = unsupported / total
        if unsupported_ratio == 0 and partial == 0:
            severity = HallucinationSeverity.NONE
        elif unsupported_ratio <= 0.1:
            severity = HallucinationSeverity.LOW
        elif unsupported_ratio <= 0.25:
            severity = HallucinationSeverity.MEDIUM
        elif unsupported_ratio <= 0.5:
            severity = HallucinationSeverity.HIGH
        else:
            severity = HallucinationSeverity.CRITICAL

        return confidence, severity


# =============================================================================
# HALLUCINATION GUARD
# =============================================================================

class HallucinationGuard:
    """
    Real-time hallucination detection for RAG pipelines.

    This guard extracts claims from generated responses and verifies each
    claim against the retrieved context to detect hallucinated content.

    Example:
        >>> guard = HallucinationGuard(llm=judge_llm)
        >>> result = await guard.detect(
        ...     response="Paris is the capital of France.",
        ...     context=["France is a country in Europe. Its capital is Paris."]
        ... )
        >>> print(f"Hallucination: {result.has_hallucination}")
        >>> print(f"Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[GuardConfig] = None
    ):
        """Initialize the hallucination guard.

        Args:
            llm: LLM provider for claims extraction and verification
            config: Guard configuration
        """
        self.llm = llm
        self.config = config or GuardConfig()

        self.extractor = ClaimsExtractor(
            llm,
            max_claims=self.config.max_claims,
            min_claim_length=self.config.min_claim_length
        )
        self.verifier = ClaimVerifier(llm, self.config)
        self.scorer = ConfidenceScorer(self.config)

    async def detect(
        self,
        response: str,
        context: List[str],
        question: Optional[str] = None
    ) -> HallucinationResult:
        """Detect hallucinations in a response.

        Args:
            response: The generated response to check
            context: List of context chunks used for generation
            question: Optional original question (for context)

        Returns:
            HallucinationResult with detection details
        """
        try:
            return await asyncio.wait_for(
                self._detect_impl(response, context, question),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning("Hallucination detection timed out")
            return HallucinationResult(
                has_hallucination=False,
                confidence=0.5,
                severity=HallucinationSeverity.LOW,
                reasoning="Detection timed out",
                metadata={"timeout": True}
            )

    async def _detect_impl(
        self,
        response: str,
        context: List[str],
        question: Optional[str] = None
    ) -> HallucinationResult:
        """Implementation of hallucination detection."""
        # Step 1: Extract claims
        claims = await self.extractor.extract(response)

        if not claims:
            return HallucinationResult(
                has_hallucination=False,
                confidence=1.0,
                severity=HallucinationSeverity.NONE,
                reasoning="No factual claims found in response",
                metadata={"no_claims": True}
            )

        # Step 2: Verify claims against context
        verified_claims = await self.verifier.verify_batch(claims, context)

        # Step 3: Calculate confidence and severity
        confidence, severity = self.scorer.calculate(verified_claims)

        # Step 4: Identify flagged claims
        flagged_claims = [
            c for c in verified_claims
            if c.verdict == ClaimVerdict.NOT_SUPPORTED
        ]

        # Include partially supported in strict mode
        if self.config.strict_mode:
            flagged_claims.extend([
                c for c in verified_claims
                if c.verdict == ClaimVerdict.PARTIALLY_SUPPORTED
            ])

        # Step 5: Determine if hallucination present
        has_hallucination = (
            confidence < self.config.confidence_threshold or
            severity in [HallucinationSeverity.HIGH, HallucinationSeverity.CRITICAL] or
            (self.config.strict_mode and len(flagged_claims) > 0)
        )

        # Calculate claim counts
        supported_count = sum(
            1 for c in verified_claims
            if c.verdict == ClaimVerdict.SUPPORTED
        )
        unsupported_count = sum(
            1 for c in verified_claims
            if c.verdict == ClaimVerdict.NOT_SUPPORTED
        )
        partial_count = sum(
            1 for c in verified_claims
            if c.verdict == ClaimVerdict.PARTIALLY_SUPPORTED
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            verified_claims, confidence, severity, has_hallucination
        )

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=confidence,
            severity=severity,
            flagged_claims=flagged_claims,
            verified_claims=verified_claims,
            total_claims=len(verified_claims),
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            partially_supported_claims=partial_count,
            reasoning=reasoning,
            metadata={
                "question": question,
                "context_count": len(context),
                "strict_mode": self.config.strict_mode
            }
        )

    def _build_reasoning(
        self,
        verified_claims: List[VerifiedClaim],
        confidence: float,
        severity: HallucinationSeverity,
        has_hallucination: bool
    ) -> str:
        """Build overall reasoning string."""
        total = len(verified_claims)
        supported = sum(1 for c in verified_claims if c.verdict == ClaimVerdict.SUPPORTED)
        unsupported = sum(1 for c in verified_claims if c.verdict == ClaimVerdict.NOT_SUPPORTED)
        partial = sum(1 for c in verified_claims if c.verdict == ClaimVerdict.PARTIALLY_SUPPORTED)

        parts = [
            f"Analyzed {total} claims.",
            f"Supported: {supported}, Partially: {partial}, Unsupported: {unsupported}.",
            f"Confidence: {confidence:.2f}, Severity: {severity.value}."
        ]

        if has_hallucination:
            parts.append("Hallucination detected - response contains unsupported claims.")
        else:
            parts.append("Response appears well-grounded in context.")

        return " ".join(parts)

    async def claim_verification(
        self,
        claims: List[str],
        context: List[str]
    ) -> List[VerifiedClaim]:
        """Verify a list of claims directly.

        This method allows direct claim verification without extraction,
        useful when claims are already known or pre-extracted.

        Args:
            claims: List of claim strings to verify
            context: Context to verify against

        Returns:
            List of VerifiedClaim objects
        """
        extracted = [
            ExtractedClaim(text=c, source_sentence=c, index=i)
            for i, c in enumerate(claims)
        ]
        return await self.verifier.verify_batch(extracted, context)

    def quick_check(
        self,
        response: str,
        context: List[str]
    ) -> Tuple[float, bool]:
        """Perform a quick heuristic check without LLM calls.

        This is useful for fast pre-filtering before full detection.

        Args:
            response: Response to check
            context: Context chunks

        Returns:
            Tuple of (overlap_score, likely_hallucinated)
        """
        if not context:
            return 0.0, True

        # Combine context
        combined_context = " ".join(context).lower()

        # Tokenize response
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        # Filter stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "it", "its", "this", "that", "these", "those",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after"
        }
        response_words = response_words - stopwords

        if not response_words:
            return 1.0, False

        # Count words found in context
        found = sum(1 for word in response_words if word in combined_context)
        overlap = found / len(response_words)

        # Heuristic threshold
        likely_hallucinated = overlap < 0.3

        return overlap, likely_hallucinated


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

class GuardedPipelineMixin:
    """Mixin to add hallucination guard to RAG pipelines.

    Example:
        class MyPipeline(RAGPipeline, GuardedPipelineMixin):
            pass

        pipeline = MyPipeline(llm=llm)
        pipeline.set_hallucination_guard(HallucinationGuard(llm=judge_llm))
    """

    _hallucination_guard: Optional[HallucinationGuard] = None
    _guard_enabled: bool = False

    def set_hallucination_guard(
        self,
        guard: HallucinationGuard,
        enabled: bool = True
    ) -> None:
        """Set the hallucination guard for this pipeline.

        Args:
            guard: HallucinationGuard instance
            enabled: Whether to enable guard by default
        """
        self._hallucination_guard = guard
        self._guard_enabled = enabled

    def enable_guard(self) -> None:
        """Enable hallucination guard."""
        if self._hallucination_guard is None:
            raise RuntimeError("Guard not set. Call set_hallucination_guard first.")
        self._guard_enabled = True

    def disable_guard(self) -> None:
        """Disable hallucination guard."""
        self._guard_enabled = False

    async def check_hallucination(
        self,
        response: str,
        context: List[str],
        question: Optional[str] = None
    ) -> HallucinationResult:
        """Check response for hallucinations.

        Args:
            response: Generated response
            context: Context used for generation
            question: Optional original question

        Returns:
            HallucinationResult

        Raises:
            RuntimeError: If guard not configured
        """
        if self._hallucination_guard is None:
            raise RuntimeError("Guard not set. Call set_hallucination_guard first.")

        return await self._hallucination_guard.detect(response, context, question)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_hallucination_guard(
    llm: LLMProvider,
    confidence_threshold: float = 0.7,
    strict_mode: bool = False,
    max_claims: int = 15,
    **kwargs
) -> HallucinationGuard:
    """Factory function to create a HallucinationGuard.

    Args:
        llm: LLM provider for detection
        confidence_threshold: Minimum confidence to pass
        strict_mode: Fail on any unsupported claim
        max_claims: Maximum claims to extract
        **kwargs: Additional config options

    Returns:
        Configured HallucinationGuard instance
    """
    config = GuardConfig(
        confidence_threshold=confidence_threshold,
        strict_mode=strict_mode,
        max_claims=max_claims,
        **kwargs
    )
    return HallucinationGuard(llm=llm, config=config)


async def quick_hallucination_check(
    response: str,
    context: List[str],
    llm: LLMProvider,
    threshold: float = 0.7
) -> Tuple[bool, float]:
    """Quick hallucination check function.

    Args:
        response: Response to check
        context: Context chunks
        llm: LLM provider
        threshold: Confidence threshold

    Returns:
        Tuple of (has_hallucination, confidence)
    """
    guard = HallucinationGuard(llm=llm, config=GuardConfig(
        confidence_threshold=threshold,
        max_claims=10,
        batch_size=10
    ))
    result = await guard.detect(response, context)
    return result.has_hallucination, result.confidence


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "HallucinationGuard",
    # Configuration
    "GuardConfig",
    # Result types
    "HallucinationResult",
    "VerifiedClaim",
    "ExtractedClaim",
    # Enums
    "ClaimVerdict",
    "HallucinationSeverity",
    # Components
    "ClaimsExtractor",
    "ClaimVerifier",
    "ConfidenceScorer",
    # Prompts
    "GuardPrompts",
    # Mixins
    "GuardedPipelineMixin",
    # Factory functions
    "create_hallucination_guard",
    "quick_hallucination_check",
]
