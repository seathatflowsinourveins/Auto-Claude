"""
Tests for Hallucination Guard Module

Tests cover:
- Claims extraction from responses
- Claim verification against context
- Confidence scoring and severity calculation
- Full hallucination detection pipeline
- Pipeline integration
"""

import asyncio
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from core.rag.hallucination_guard import (
    HallucinationGuard,
    GuardConfig,
    HallucinationResult,
    VerifiedClaim,
    ExtractedClaim,
    ClaimVerdict,
    HallucinationSeverity,
    ClaimsExtractor,
    ClaimVerifier,
    ConfidenceScorer,
    create_hallucination_guard,
    quick_hallucination_check,
)


# =============================================================================
# FIXTURES
# =============================================================================

class MockLLM:
    """Mock LLM provider for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "CLAIM 1: Default claim"


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MockLLM()


@pytest.fixture
def guard_config():
    """Create a test guard configuration."""
    return GuardConfig(
        confidence_threshold=0.7,
        max_claims=10,
        batch_size=3,
        strict_mode=False,
    )


@pytest.fixture
def sample_response():
    """Sample response text for testing."""
    return (
        "Paris is the capital of France. The Eiffel Tower is located in Paris. "
        "The tower was built in 1889 for the World's Fair. It stands 324 meters tall."
    )


@pytest.fixture
def sample_context():
    """Sample context chunks for testing."""
    return [
        "France is a country in Western Europe. Its capital city is Paris.",
        "The Eiffel Tower is a famous landmark in Paris, France. "
        "It was constructed in 1889 as the entrance arch for the World's Fair.",
        "The Eiffel Tower stands at 324 meters (1,063 feet) tall.",
    ]


# =============================================================================
# CLAIMS EXTRACTOR TESTS
# =============================================================================

class TestClaimsExtractor:
    """Tests for ClaimsExtractor."""

    @pytest.mark.asyncio
    async def test_extract_claims_success(self, sample_response):
        """Test successful claims extraction."""
        mock_llm = MockLLM(responses=[
            "CLAIM 1: Paris is the capital of France\n"
            "CLAIM 2: The Eiffel Tower is located in Paris\n"
            "CLAIM 3: The tower was built in 1889"
        ])

        extractor = ClaimsExtractor(mock_llm, max_claims=10)
        claims = await extractor.extract(sample_response)

        assert len(claims) == 3
        assert claims[0].text == "Paris is the capital of France"
        assert claims[1].text == "The Eiffel Tower is located in Paris"
        assert claims[2].text == "The tower was built in 1889"

    @pytest.mark.asyncio
    async def test_extract_claims_max_limit(self):
        """Test claims extraction respects max limit."""
        mock_llm = MockLLM(responses=[
            "CLAIM 1: Claim one\n"
            "CLAIM 2: Claim two\n"
            "CLAIM 3: Claim three\n"
            "CLAIM 4: Claim four\n"
            "CLAIM 5: Claim five"
        ])

        extractor = ClaimsExtractor(mock_llm, max_claims=3)
        claims = await extractor.extract("Test response")

        assert len(claims) == 3

    @pytest.mark.asyncio
    async def test_extract_claims_fallback(self, sample_response):
        """Test fallback extraction when LLM fails."""
        mock_llm = MockLLM()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        extractor = ClaimsExtractor(mock_llm, max_claims=10)
        claims = await extractor.extract(sample_response)

        # Should use fallback sentence-based extraction
        assert len(claims) > 0


# =============================================================================
# CLAIM VERIFIER TESTS
# =============================================================================

class TestClaimVerifier:
    """Tests for ClaimVerifier."""

    @pytest.mark.asyncio
    async def test_verify_supported_claim(self, sample_context, guard_config):
        """Test verification of a supported claim."""
        mock_llm = MockLLM(responses=[
            "VERDICT: SUPPORTED\n"
            "CONFIDENCE: 0.95\n"
            "EVIDENCE: France is a country in Western Europe. Its capital city is Paris.\n"
            "REASONING: The context explicitly states Paris is the capital of France."
        ])

        verifier = ClaimVerifier(mock_llm, guard_config)
        claim = ExtractedClaim(
            text="Paris is the capital of France",
            source_sentence="Paris is the capital of France.",
            index=1
        )

        result = await verifier.verify_single(claim, sample_context)

        assert result.verdict == ClaimVerdict.SUPPORTED
        assert result.confidence >= 0.9
        assert result.supporting_evidence is not None

    @pytest.mark.asyncio
    async def test_verify_unsupported_claim(self, sample_context, guard_config):
        """Test verification of an unsupported claim."""
        mock_llm = MockLLM(responses=[
            "VERDICT: NOT_SUPPORTED\n"
            "CONFIDENCE: 0.1\n"
            "EVIDENCE: N/A\n"
            "REASONING: The context does not mention anything about London."
        ])

        verifier = ClaimVerifier(mock_llm, guard_config)
        claim = ExtractedClaim(
            text="London is the capital of France",
            source_sentence="London is the capital of France.",
            index=1
        )

        result = await verifier.verify_single(claim, sample_context)

        assert result.verdict == ClaimVerdict.NOT_SUPPORTED
        assert result.confidence <= 0.2

    @pytest.mark.asyncio
    async def test_verify_batch(self, sample_context, guard_config):
        """Test batch verification of multiple claims."""
        mock_llm = MockLLM(responses=[
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.9\nEVIDENCE: test\nREASONING: ok",
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.85\nEVIDENCE: test\nREASONING: ok",
            "VERDICT: NOT_SUPPORTED\nCONFIDENCE: 0.2\nEVIDENCE: N/A\nREASONING: not found",
        ])

        verifier = ClaimVerifier(mock_llm, guard_config)
        claims = [
            ExtractedClaim(text="Claim 1", source_sentence="Claim 1", index=1),
            ExtractedClaim(text="Claim 2", source_sentence="Claim 2", index=2),
            ExtractedClaim(text="Claim 3", source_sentence="Claim 3", index=3),
        ]

        results = await verifier.verify_batch(claims, sample_context)

        assert len(results) == 3
        assert results[0].verdict == ClaimVerdict.SUPPORTED
        assert results[2].verdict == ClaimVerdict.NOT_SUPPORTED


# =============================================================================
# CONFIDENCE SCORER TESTS
# =============================================================================

class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""

    def test_all_supported_claims(self, guard_config):
        """Test confidence with all supported claims."""
        scorer = ConfidenceScorer(guard_config)
        claims = [
            VerifiedClaim(
                text="Claim 1",
                source_sentence="Claim 1",
                verdict=ClaimVerdict.SUPPORTED,
                confidence=0.9
            ),
            VerifiedClaim(
                text="Claim 2",
                source_sentence="Claim 2",
                verdict=ClaimVerdict.SUPPORTED,
                confidence=0.95
            ),
        ]

        confidence, severity = scorer.calculate(claims)

        assert confidence == 1.0
        assert severity == HallucinationSeverity.NONE

    def test_mixed_claims(self, guard_config):
        """Test confidence with mixed claim verdicts."""
        scorer = ConfidenceScorer(guard_config)
        claims = [
            VerifiedClaim(
                text="Claim 1",
                source_sentence="Claim 1",
                verdict=ClaimVerdict.SUPPORTED,
                confidence=0.9
            ),
            VerifiedClaim(
                text="Claim 2",
                source_sentence="Claim 2",
                verdict=ClaimVerdict.NOT_SUPPORTED,
                confidence=0.1
            ),
        ]

        confidence, severity = scorer.calculate(claims)

        assert 0.4 < confidence < 0.6  # ~0.5 for 1 supported, 1 unsupported
        assert severity in [HallucinationSeverity.HIGH, HallucinationSeverity.CRITICAL]

    def test_no_claims(self, guard_config):
        """Test confidence with no claims."""
        scorer = ConfidenceScorer(guard_config)

        confidence, severity = scorer.calculate([])

        assert confidence == 1.0
        assert severity == HallucinationSeverity.NONE


# =============================================================================
# HALLUCINATION GUARD TESTS
# =============================================================================

class TestHallucinationGuard:
    """Tests for HallucinationGuard."""

    @pytest.mark.asyncio
    async def test_detect_no_hallucination(
        self, sample_response, sample_context, guard_config
    ):
        """Test detection with no hallucination."""
        mock_llm = MockLLM(responses=[
            # Claims extraction
            "CLAIM 1: Paris is the capital of France\n"
            "CLAIM 2: The Eiffel Tower is in Paris",
            # Verification for claim 1
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.95\nEVIDENCE: text\nREASONING: ok",
            # Verification for claim 2
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.9\nEVIDENCE: text\nREASONING: ok",
        ])

        guard = HallucinationGuard(mock_llm, guard_config)
        result = await guard.detect(sample_response, sample_context)

        assert result.has_hallucination is False
        assert result.confidence >= 0.7
        assert result.severity == HallucinationSeverity.NONE
        assert len(result.flagged_claims) == 0

    @pytest.mark.asyncio
    async def test_detect_hallucination(
        self, sample_context, guard_config
    ):
        """Test detection with hallucination present."""
        response = "Paris is the capital of France. The Eiffel Tower was built by aliens."

        mock_llm = MockLLM(responses=[
            # Claims extraction
            "CLAIM 1: Paris is the capital of France\n"
            "CLAIM 2: The Eiffel Tower was built by aliens",
            # Verification for claim 1
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.95\nEVIDENCE: text\nREASONING: ok",
            # Verification for claim 2 (hallucinated)
            "VERDICT: NOT_SUPPORTED\nCONFIDENCE: 0.1\nEVIDENCE: N/A\n"
            "REASONING: The context says it was built in 1889, not by aliens.",
        ])

        guard = HallucinationGuard(mock_llm, guard_config)
        result = await guard.detect(response, sample_context)

        assert result.has_hallucination is True
        assert len(result.flagged_claims) > 0
        assert result.unsupported_claims > 0

    @pytest.mark.asyncio
    async def test_quick_check(self, sample_response, sample_context):
        """Test quick heuristic check."""
        guard = HallucinationGuard(MockLLM(), GuardConfig())

        overlap, likely_hallucinated = guard.quick_check(
            sample_response, sample_context
        )

        # Should have good overlap since response terms are in context
        assert overlap > 0.3
        assert likely_hallucinated is False

    @pytest.mark.asyncio
    async def test_claim_verification_direct(self, sample_context, guard_config):
        """Test direct claim verification."""
        mock_llm = MockLLM(responses=[
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.9\nEVIDENCE: text\nREASONING: ok",
            "VERDICT: NOT_SUPPORTED\nCONFIDENCE: 0.2\nEVIDENCE: N/A\nREASONING: no",
        ])

        guard = HallucinationGuard(mock_llm, guard_config)

        claims = [
            "Paris is the capital of France",
            "The Eiffel Tower is on the moon",
        ]

        results = await guard.claim_verification(claims, sample_context)

        assert len(results) == 2
        assert results[0].verdict == ClaimVerdict.SUPPORTED
        assert results[1].verdict == ClaimVerdict.NOT_SUPPORTED


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_hallucination_guard(self):
        """Test guard creation with factory function."""
        mock_llm = MockLLM()

        guard = create_hallucination_guard(
            llm=mock_llm,
            confidence_threshold=0.8,
            strict_mode=True,
            max_claims=5,
        )

        assert guard is not None
        assert guard.config.confidence_threshold == 0.8
        assert guard.config.strict_mode is True
        assert guard.config.max_claims == 5

    @pytest.mark.asyncio
    async def test_quick_hallucination_check_function(self, sample_response, sample_context):
        """Test quick check utility function."""
        mock_llm = MockLLM(responses=[
            "CLAIM 1: Paris is the capital of France",
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.9\nEVIDENCE: text\nREASONING: ok",
        ])

        has_hallucination, confidence = await quick_hallucination_check(
            response=sample_response,
            context=sample_context,
            llm=mock_llm,
            threshold=0.7,
        )

        assert isinstance(has_hallucination, bool)
        assert 0 <= confidence <= 1


# =============================================================================
# RESULT SERIALIZATION TESTS
# =============================================================================

class TestHallucinationResult:
    """Tests for HallucinationResult serialization."""

    def test_to_dict(self):
        """Test result serialization to dictionary."""
        result = HallucinationResult(
            has_hallucination=True,
            confidence=0.65,
            severity=HallucinationSeverity.MEDIUM,
            flagged_claims=[
                VerifiedClaim(
                    text="Hallucinated claim",
                    source_sentence="Hallucinated claim",
                    verdict=ClaimVerdict.NOT_SUPPORTED,
                    confidence=0.1,
                    reasoning="Not found in context",
                )
            ],
            total_claims=3,
            supported_claims=2,
            unsupported_claims=1,
            reasoning="One claim is not supported by context.",
        )

        result_dict = result.to_dict()

        assert result_dict["has_hallucination"] is True
        assert result_dict["confidence"] == 0.65
        assert result_dict["severity"] == "medium"
        assert len(result_dict["flagged_claims"]) == 1
        assert result_dict["total_claims"] == 3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPipelineIntegration:
    """Tests for pipeline integration."""

    @pytest.mark.asyncio
    async def test_guarded_pipeline_mixin(self):
        """Test GuardedPipelineMixin functionality."""
        from core.rag.hallucination_guard import GuardedPipelineMixin

        class MockPipeline(GuardedPipelineMixin):
            pass

        pipeline = MockPipeline()
        mock_llm = MockLLM(responses=[
            "CLAIM 1: Test claim",
            "VERDICT: SUPPORTED\nCONFIDENCE: 0.9\nEVIDENCE: text\nREASONING: ok",
        ])

        guard = HallucinationGuard(mock_llm, GuardConfig())
        pipeline.set_hallucination_guard(guard)

        assert pipeline._guard_enabled is True

        result = await pipeline.check_hallucination(
            response="Test response",
            context=["Test context"],
            question="Test question",
        )

        assert isinstance(result, HallucinationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
