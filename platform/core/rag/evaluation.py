"""
RAG Evaluation: Ragas-Style Metrics for RAG Pipeline Quality Assessment

This module provides comprehensive evaluation metrics for Retrieval-Augmented
Generation pipelines, implementing the Ragas evaluation framework patterns.

Key Features:
- Context Metrics: Precision, Recall, Relevancy
- Answer Metrics: Relevancy, Correctness, Faithfulness
- Hallucination Detection: Claims extraction and verification
- End-to-End Metrics: NDCG@k, MRR, F1
- LLM-as-Judge: Evaluation prompts with structured output

Architecture:
    Questions + Contexts + Answers + Ground Truth
                      |
                      v
    +------------------------------------------+
    |            RAGEvaluator                  |
    |------------------------------------------|
    | - Context Metrics (precision/recall)     |
    | - Answer Metrics (relevancy/faithfulness)|
    | - Hallucination Detection                |
    | - Ranking Metrics (NDCG/MRR)             |
    +------------------------------------------+
                      |
                      v
              EvaluationResult
              (scores, details, recommendations)

Reference: https://docs.ragas.io/en/latest/concepts/metrics/

Integration:
    from core.rag.evaluation import RAGEvaluator, EvaluationConfig

    evaluator = RAGEvaluator(llm=judge_llm)
    results = await evaluator.evaluate(
        questions=["What is RAG?"],
        retrieved_contexts=[["Context about RAG..."]],
        generated_answers=["RAG is..."],
        ground_truth=["RAG stands for..."]  # optional
    )

    print(f"Faithfulness: {results.faithfulness}")
    print(f"Answer Relevancy: {results.answer_relevancy}")
    print(f"Context Precision: {results.context_precision}")
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS AND TYPES
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers used in evaluation."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts to vectors."""
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation.

    Attributes:
        faithfulness_threshold: Min score to pass faithfulness (default: 0.7)
        answer_relevancy_threshold: Min score to pass answer relevancy (default: 0.7)
        context_precision_threshold: Min score to pass context precision (default: 0.5)
        context_recall_threshold: Min score to pass context recall (default: 0.5)
        hallucination_threshold: Max hallucination score to pass (default: 0.3)
        ndcg_k: Number of top results for NDCG calculation (default: 10)
        batch_size: Batch size for parallel evaluation (default: 5)
        max_claims_per_answer: Max claims to extract from answer (default: 10)
        temperature: LLM temperature for evaluation (default: 0.0)
        enable_caching: Cache intermediate LLM calls (default: True)
    """
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    context_precision_threshold: float = 0.5
    context_recall_threshold: float = 0.5
    hallucination_threshold: float = 0.3
    ndcg_k: int = 10
    batch_size: int = 5
    max_claims_per_answer: int = 10
    temperature: float = 0.0
    enable_caching: bool = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Claim:
    """A factual claim extracted from an answer."""
    text: str
    source_sentence: str
    confidence: float = 1.0
    is_supported: Optional[bool] = None
    supporting_context: Optional[str] = None
    verification_reasoning: str = ""


@dataclass
class ContextRelevance:
    """Relevance assessment for a single context chunk."""
    context: str
    relevance_score: float
    is_relevant: bool
    reasoning: str = ""


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class SampleEvaluation:
    """Evaluation result for a single sample (question/answer pair)."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    # Metric scores
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0
    answer_correctness: float = 0.0
    hallucination_score: float = 0.0

    # Detailed results
    claims: List[Claim] = field(default_factory=list)
    context_relevances: List[ContextRelevance] = field(default_factory=list)

    # Pass/fail
    passed: bool = False
    failed_metrics: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Aggregated evaluation result across all samples."""
    # Aggregate scores (averages)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0
    answer_correctness: float = 0.0
    hallucination_score: float = 0.0

    # Ranking metrics
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    f1_score: float = 0.0

    # Per-sample results
    samples: List[SampleEvaluation] = field(default_factory=list)

    # Summary
    total_samples: int = 0
    passed_samples: int = 0
    failed_samples: int = 0
    overall_passed: bool = False

    # Metadata
    config: Optional[EvaluationConfig] = None
    duration_ms: int = 0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scores": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "context_relevancy": self.context_relevancy,
                "answer_correctness": self.answer_correctness,
                "hallucination_score": self.hallucination_score,
                "ndcg_at_k": self.ndcg_at_k,
                "mrr": self.mrr,
                "f1_score": self.f1_score,
            },
            "summary": {
                "total_samples": self.total_samples,
                "passed_samples": self.passed_samples,
                "failed_samples": self.failed_samples,
                "overall_passed": self.overall_passed,
            },
            "duration_ms": self.duration_ms,
            "recommendations": self.recommendations,
        }


# =============================================================================
# EVALUATION PROMPTS (LLM-AS-JUDGE)
# =============================================================================

class EvaluationPrompts:
    """Prompt templates for LLM-based evaluation."""

    EXTRACT_CLAIMS = """Extract all factual claims from the following answer.
A claim is a single, verifiable statement of fact.

Answer:
{answer}

Instructions:
1. Break down the answer into individual factual claims
2. Each claim should be self-contained and verifiable
3. Exclude opinions, hedged statements, and meta-commentary
4. Number each claim

Output format:
CLAIM 1: [first factual claim]
CLAIM 2: [second factual claim]
...

Claims:"""

    VERIFY_CLAIM = """Determine if the following claim is supported by the provided context.

Claim: {claim}

Context:
{context}

Instructions:
1. Check if the claim can be verified from the context
2. Consider both explicit and reasonably implied information
3. Be strict: if the context doesn't support the claim, mark as NOT_SUPPORTED

Output:
VERDICT: [SUPPORTED/NOT_SUPPORTED/PARTIALLY_SUPPORTED]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
SUPPORTING_TEXT: [quote from context if supported, or "N/A"]"""

    CONTEXT_RELEVANCE = """Rate how relevant this context is to answering the question.

Question: {question}

Context:
{context}

Instructions:
1. Assess if the context contains information useful for answering
2. Consider both direct and indirect relevance
3. Score from 0.0 (completely irrelevant) to 1.0 (highly relevant)

Output:
RELEVANCE: [0.0-1.0]
IS_RELEVANT: [YES/NO] (YES if score >= 0.5)
REASONING: [brief explanation]"""

    ANSWER_RELEVANCY = """Rate how well this answer addresses the question.

Question: {question}

Answer: {answer}

Instructions:
1. Does the answer directly address what was asked?
2. Is the answer complete and informative?
3. Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant)

Output:
RELEVANCE: [0.0-1.0]
ADDRESSES_QUESTION: [FULLY/PARTIALLY/NOT_AT_ALL]
REASONING: [brief explanation]"""

    ANSWER_CORRECTNESS = """Compare the generated answer to the ground truth.

Question: {question}

Generated Answer: {answer}

Ground Truth: {ground_truth}

Instructions:
1. Check factual accuracy against ground truth
2. Consider semantic similarity, not just exact match
3. Penalize contradictions heavily
4. Score from 0.0 (completely wrong) to 1.0 (perfectly correct)

Output:
CORRECTNESS: [0.0-1.0]
FACTUAL_OVERLAP: [HIGH/MEDIUM/LOW/NONE]
CONTRADICTIONS: [list any contradictions, or "None"]
REASONING: [brief explanation]"""

    DETECT_HALLUCINATIONS = """Identify any hallucinated (fabricated) information in the answer.

Question: {question}

Answer: {answer}

Available Context:
{context}

Instructions:
1. Identify any claims in the answer NOT supported by the context
2. Check for fabricated facts, statistics, or citations
3. Note any information that goes beyond the context

Output:
HALLUCINATION_DETECTED: [YES/NO]
HALLUCINATED_CLAIMS: [list hallucinated claims, or "None"]
HALLUCINATION_SEVERITY: [NONE/LOW/MEDIUM/HIGH]
REASONING: [brief explanation]"""

    CONTEXT_PRECISION = """Evaluate if the retrieved contexts are ranked by relevance.

Question: {question}

Contexts (in retrieval order):
{contexts}

Instructions:
1. Check if more relevant contexts appear earlier
2. Assess the precision of the top-k results
3. Consider if irrelevant contexts are mixed with relevant ones

Output:
PRECISION_SCORE: [0.0-1.0]
RANKING_QUALITY: [EXCELLENT/GOOD/FAIR/POOR]
REASONING: [brief explanation]"""

    CONTEXT_RECALL = """Evaluate if the contexts contain all information needed to answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Contexts:
{contexts}

Instructions:
1. Identify key facts needed from ground truth
2. Check if contexts contain those facts
3. Score based on coverage of required information

Output:
RECALL_SCORE: [0.0-1.0]
COVERAGE: [COMPLETE/PARTIAL/MINIMAL/NONE]
MISSING_INFORMATION: [list what's missing, or "None"]
REASONING: [brief explanation]"""


# =============================================================================
# METRIC CALCULATORS
# =============================================================================

class ClaimsExtractor:
    """Extracts factual claims from answers using LLM."""

    def __init__(self, llm: LLMProvider, max_claims: int = 10):
        self.llm = llm
        self.max_claims = max_claims
        self.prompts = EvaluationPrompts()

    async def extract(self, answer: str) -> List[Claim]:
        """Extract factual claims from an answer."""
        prompt = self.prompts.EXTRACT_CLAIMS.format(answer=answer)

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=800,
                temperature=0.0
            )
            return self._parse_claims(response, answer)
        except Exception as e:
            logger.warning(f"Claims extraction failed: {e}")
            # Fallback: split by sentences
            return self._fallback_extraction(answer)

    def _parse_claims(self, response: str, original_answer: str) -> List[Claim]:
        """Parse LLM response into Claim objects."""
        claims = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "CLAIM N:" pattern
            match = re.match(r'^CLAIM\s*\d+\s*:\s*(.+)$', line, re.IGNORECASE)
            if match:
                claim_text = match.group(1).strip()
                if claim_text and len(claim_text) > 10:
                    claims.append(Claim(
                        text=claim_text,
                        source_sentence=claim_text,  # Will be refined later
                        confidence=1.0
                    ))

            if len(claims) >= self.max_claims:
                break

        return claims

    def _fallback_extraction(self, answer: str) -> List[Claim]:
        """Fallback extraction using sentence splitting."""
        import re
        sentences = re.split(r'[.!?]+', answer)
        claims = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:  # Skip very short fragments
                claims.append(Claim(
                    text=sent,
                    source_sentence=sent,
                    confidence=0.8
                ))

            if len(claims) >= self.max_claims:
                break

        return claims


class ClaimsVerifier:
    """Verifies claims against context using LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def verify(
        self,
        claim: Claim,
        contexts: List[str]
    ) -> Claim:
        """Verify a single claim against provided contexts."""
        combined_context = "\n\n".join(contexts)

        prompt = self.prompts.VERIFY_CLAIM.format(
            claim=claim.text,
            context=combined_context[:4000]  # Truncate for token limit
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.0
            )
            return self._parse_verification(claim, response)
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            claim.is_supported = None
            claim.verification_reasoning = f"Verification error: {e}"
            return claim

    def _parse_verification(self, claim: Claim, response: str) -> Claim:
        """Parse verification response."""
        response_upper = response.upper()

        # Parse verdict
        if "VERDICT: SUPPORTED" in response_upper or "VERDICT:SUPPORTED" in response_upper:
            claim.is_supported = True
        elif "VERDICT: NOT_SUPPORTED" in response_upper or "NOT SUPPORTED" in response_upper:
            claim.is_supported = False
        elif "PARTIALLY" in response_upper:
            claim.is_supported = True  # Count partial as supported
            claim.confidence = 0.6
        else:
            claim.is_supported = False

        # Parse confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
        if conf_match:
            try:
                claim.confidence = float(conf_match.group(1))
            except ValueError:
                pass

        # Parse reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?:SUPPORTING_TEXT:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            claim.verification_reasoning = reasoning_match.group(1).strip()

        # Parse supporting text
        support_match = re.search(
            r'SUPPORTING_TEXT:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if support_match:
            text = support_match.group(1).strip()
            if text.upper() != "N/A":
                claim.supporting_context = text

        return claim

    async def verify_batch(
        self,
        claims: List[Claim],
        contexts: List[str],
        batch_size: int = 5
    ) -> List[Claim]:
        """Verify multiple claims in parallel batches."""
        verified_claims = []

        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            tasks = [self.verify(claim, contexts) for claim in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Claim):
                    verified_claims.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Claim verification failed: {result}")

        return verified_claims


class FaithfulnessCalculator:
    """Calculates faithfulness score based on claim verification."""

    def __init__(self, llm: LLMProvider, config: EvaluationConfig):
        self.extractor = ClaimsExtractor(llm, config.max_claims_per_answer)
        self.verifier = ClaimsVerifier(llm)
        self.config = config

    async def calculate(
        self,
        answer: str,
        contexts: List[str]
    ) -> Tuple[float, List[Claim]]:
        """Calculate faithfulness score.

        Faithfulness = (# supported claims) / (# total claims)

        Returns:
            Tuple of (score, list of claims)
        """
        # Extract claims
        claims = await self.extractor.extract(answer)

        if not claims:
            return 1.0, []  # No claims = perfect faithfulness

        # Verify claims
        verified_claims = await self.verifier.verify_batch(
            claims, contexts, self.config.batch_size
        )

        # Calculate score
        supported_count = sum(
            1 for c in verified_claims
            if c.is_supported is True
        )
        total_count = len(verified_claims)

        score = supported_count / total_count if total_count > 0 else 0.0

        return score, verified_claims


class ContextRelevanceCalculator:
    """Calculates context relevance metrics using LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def calculate_single(
        self,
        question: str,
        context: str
    ) -> ContextRelevance:
        """Calculate relevance of a single context chunk."""
        prompt = self.prompts.CONTEXT_RELEVANCE.format(
            question=question,
            context=context[:2000]
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=200,
                temperature=0.0
            )
            return self._parse_relevance(context, response)
        except Exception as e:
            logger.warning(f"Context relevance calculation failed: {e}")
            return ContextRelevance(
                context=context,
                relevance_score=0.5,
                is_relevant=False,
                reasoning=f"Error: {e}"
            )

    def _parse_relevance(self, context: str, response: str) -> ContextRelevance:
        """Parse relevance response."""
        # Parse score
        score = 0.5
        score_match = re.search(r'RELEVANCE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Parse is_relevant
        is_relevant = score >= 0.5
        if "IS_RELEVANT: YES" in response.upper():
            is_relevant = True
        elif "IS_RELEVANT: NO" in response.upper():
            is_relevant = False

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return ContextRelevance(
            context=context,
            relevance_score=score,
            is_relevant=is_relevant,
            reasoning=reasoning
        )

    async def calculate_batch(
        self,
        question: str,
        contexts: List[str]
    ) -> List[ContextRelevance]:
        """Calculate relevance for multiple contexts."""
        tasks = [
            self.calculate_single(question, ctx)
            for ctx in contexts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        relevances = []
        for result in results:
            if isinstance(result, ContextRelevance):
                relevances.append(result)
            else:
                logger.warning(f"Context relevance failed: {result}")

        return relevances


class AnswerRelevancyCalculator:
    """Calculates answer relevancy using LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def calculate(
        self,
        question: str,
        answer: str
    ) -> Tuple[float, str]:
        """Calculate answer relevancy score.

        Returns:
            Tuple of (score, reasoning)
        """
        prompt = self.prompts.ANSWER_RELEVANCY.format(
            question=question,
            answer=answer
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.0
            )
            return self._parse_relevancy(response)
        except Exception as e:
            logger.warning(f"Answer relevancy calculation failed: {e}")
            return 0.5, f"Error: {e}"

    def _parse_relevancy(self, response: str) -> Tuple[float, str]:
        """Parse relevancy response."""
        # Parse score
        score = 0.5
        score_match = re.search(r'RELEVANCE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Adjust based on addresses_question
        response_upper = response.upper()
        if "ADDRESSES_QUESTION: FULLY" in response_upper:
            score = max(score, 0.8)
        elif "ADDRESSES_QUESTION: NOT_AT_ALL" in response_upper:
            score = min(score, 0.3)

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning


class AnswerCorrectnessCalculator:
    """Calculates answer correctness against ground truth."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def calculate(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> Tuple[float, str]:
        """Calculate answer correctness score.

        Returns:
            Tuple of (score, reasoning)
        """
        prompt = self.prompts.ANSWER_CORRECTNESS.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=400,
                temperature=0.0
            )
            return self._parse_correctness(response)
        except Exception as e:
            logger.warning(f"Answer correctness calculation failed: {e}")
            return 0.5, f"Error: {e}"

    def _parse_correctness(self, response: str) -> Tuple[float, str]:
        """Parse correctness response."""
        # Parse score
        score = 0.5
        score_match = re.search(r'CORRECTNESS:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Adjust based on overlap
        response_upper = response.upper()
        if "FACTUAL_OVERLAP: HIGH" in response_upper:
            score = max(score, 0.7)
        elif "FACTUAL_OVERLAP: NONE" in response_upper:
            score = min(score, 0.3)

        # Heavy penalty for contradictions
        if "CONTRADICTIONS:" in response_upper and "NONE" not in response_upper.split("CONTRADICTIONS:")[-1][:50]:
            score *= 0.5

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning


class HallucinationDetector:
    """Detects hallucinations in generated answers."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def detect(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Tuple[float, List[str], str]:
        """Detect hallucinations in an answer.

        Returns:
            Tuple of (hallucination_score, hallucinated_claims, reasoning)
            hallucination_score: 0.0 = no hallucinations, 1.0 = fully hallucinated
        """
        combined_context = "\n\n".join(contexts)

        prompt = self.prompts.DETECT_HALLUCINATIONS.format(
            question=question,
            answer=answer,
            context=combined_context[:4000]
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=500,
                temperature=0.0
            )
            return self._parse_hallucination(response)
        except Exception as e:
            logger.warning(f"Hallucination detection failed: {e}")
            return 0.5, [], f"Error: {e}"

    def _parse_hallucination(
        self,
        response: str
    ) -> Tuple[float, List[str], str]:
        """Parse hallucination detection response."""
        response_upper = response.upper()

        # Parse detection
        detected = "HALLUCINATION_DETECTED: YES" in response_upper

        # Parse severity to score
        severity_map = {
            "NONE": 0.0,
            "LOW": 0.25,
            "MEDIUM": 0.5,
            "HIGH": 0.8
        }
        score = 0.0
        for severity, value in severity_map.items():
            if f"HALLUCINATION_SEVERITY: {severity}" in response_upper:
                score = value
                break

        if detected and score == 0.0:
            score = 0.3  # Default if detected but no severity

        # Parse hallucinated claims
        claims = []
        claims_match = re.search(
            r'HALLUCINATED_CLAIMS:\s*(.+?)(?:HALLUCINATION_SEVERITY:|REASONING:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if claims_match:
            claims_text = claims_match.group(1).strip()
            if claims_text.upper() != "NONE":
                # Split by common delimiters
                for delim in ["\n-", "\n*", "\n", ";", ","]:
                    if delim in claims_text:
                        claims = [
                            c.strip().lstrip("-*").strip()
                            for c in claims_text.split(delim)
                            if c.strip() and c.strip().upper() != "NONE"
                        ]
                        break
                if not claims and claims_text:
                    claims = [claims_text]

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, claims, reasoning


class ContextPrecisionCalculator:
    """Calculates context precision using LLM evaluation."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def calculate(
        self,
        question: str,
        contexts: List[str]
    ) -> Tuple[float, str]:
        """Calculate context precision score.

        Context precision measures if relevant contexts are ranked higher.

        Returns:
            Tuple of (score, reasoning)
        """
        # Format contexts with ranking
        formatted_contexts = "\n\n".join([
            f"[Rank {i+1}]\n{ctx[:500]}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = self.prompts.CONTEXT_PRECISION.format(
            question=question,
            contexts=formatted_contexts
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=300,
                temperature=0.0
            )
            return self._parse_precision(response)
        except Exception as e:
            logger.warning(f"Context precision calculation failed: {e}")
            return 0.5, f"Error: {e}"

    def _parse_precision(self, response: str) -> Tuple[float, str]:
        """Parse precision response."""
        # Parse score
        score = 0.5
        score_match = re.search(r'PRECISION_SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Adjust based on ranking quality
        response_upper = response.upper()
        if "RANKING_QUALITY: EXCELLENT" in response_upper:
            score = max(score, 0.9)
        elif "RANKING_QUALITY: POOR" in response_upper:
            score = min(score, 0.3)

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning


class ContextRecallCalculator:
    """Calculates context recall against ground truth."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.prompts = EvaluationPrompts()

    async def calculate(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str
    ) -> Tuple[float, str]:
        """Calculate context recall score.

        Context recall measures if contexts contain information needed for ground truth.

        Returns:
            Tuple of (score, reasoning)
        """
        formatted_contexts = "\n\n".join([
            f"[Context {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = self.prompts.CONTEXT_RECALL.format(
            question=question,
            ground_truth=ground_truth,
            contexts=formatted_contexts[:4000]
        )

        try:
            response = await self.llm.generate(
                prompt,
                max_tokens=400,
                temperature=0.0
            )
            return self._parse_recall(response)
        except Exception as e:
            logger.warning(f"Context recall calculation failed: {e}")
            return 0.5, f"Error: {e}"

    def _parse_recall(self, response: str) -> Tuple[float, str]:
        """Parse recall response."""
        # Parse score
        score = 0.5
        score_match = re.search(r'RECALL_SCORE:\s*([\d.]+)', response, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Adjust based on coverage
        response_upper = response.upper()
        if "COVERAGE: COMPLETE" in response_upper:
            score = max(score, 0.9)
        elif "COVERAGE: NONE" in response_upper:
            score = min(score, 0.1)

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r'REASONING:\s*(.+)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning


# =============================================================================
# RANKING METRICS (NON-LLM)
# =============================================================================

class RankingMetrics:
    """Calculates ranking metrics like NDCG, MRR, F1."""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain at k."""
        relevances = relevances[:k]
        if not relevances:
            return 0.0

        dcg = relevances[0]
        for i, rel in enumerate(relevances[1:], start=2):
            dcg += rel / math.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k.

        Args:
            relevances: List of relevance scores (0.0-1.0) in retrieval order
            k: Number of top results to consider

        Returns:
            NDCG score (0.0-1.0)
        """
        dcg = RankingMetrics.dcg_at_k(relevances, k)

        # Ideal DCG (perfect ranking)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = RankingMetrics.dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr(relevances: List[float], threshold: float = 0.5) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            relevances: List of relevance scores in retrieval order
            threshold: Score above which a result is considered relevant

        Returns:
            MRR score (0.0-1.0)
        """
        for i, rel in enumerate(relevances):
            if rel >= threshold:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def precision_at_k(relevances: List[float], k: int, threshold: float = 0.5) -> float:
        """Calculate Precision at k."""
        top_k = relevances[:k]
        if not top_k:
            return 0.0

        relevant_count = sum(1 for rel in top_k if rel >= threshold)
        return relevant_count / len(top_k)

    @staticmethod
    def recall_at_k(
        relevances: List[float],
        k: int,
        total_relevant: int,
        threshold: float = 0.5
    ) -> float:
        """Calculate Recall at k."""
        if total_relevant == 0:
            return 1.0  # No relevant items to find

        top_k = relevances[:k]
        relevant_count = sum(1 for rel in top_k if rel >= threshold)
        return relevant_count / total_relevant

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def token_overlap_f1(answer: str, ground_truth: str) -> float:
        """Calculate F1 score based on token overlap.

        Args:
            answer: Generated answer
            ground_truth: Expected answer

        Returns:
            F1 score (0.0-1.0)
        """
        # Simple tokenization
        def tokenize(text: str) -> set:
            import re
            tokens = re.findall(r'\b\w+\b', text.lower())
            return set(tokens)

        answer_tokens = tokenize(answer)
        truth_tokens = tokenize(ground_truth)

        if not truth_tokens:
            return 1.0 if not answer_tokens else 0.0

        if not answer_tokens:
            return 0.0

        common = answer_tokens & truth_tokens

        precision = len(common) / len(answer_tokens)
        recall = len(common) / len(truth_tokens)

        return RankingMetrics.f1_score(precision, recall)


# =============================================================================
# MAIN EVALUATOR
# =============================================================================

class RAGEvaluator:
    """
    Comprehensive RAG evaluation using Ragas-style metrics.

    Evaluates RAG pipeline quality across multiple dimensions:
    - Faithfulness: Are answers grounded in context?
    - Answer Relevancy: Do answers address the question?
    - Context Precision: Are relevant contexts ranked higher?
    - Context Recall: Do contexts contain needed information?
    - Hallucination: Is fabricated information present?

    Example:
        >>> from core.rag.evaluation import RAGEvaluator, EvaluationConfig
        >>>
        >>> config = EvaluationConfig(faithfulness_threshold=0.7)
        >>> evaluator = RAGEvaluator(llm=judge_llm, config=config)
        >>>
        >>> results = await evaluator.evaluate(
        ...     questions=["What is RAG?"],
        ...     retrieved_contexts=[["RAG is a technique..."]],
        ...     generated_answers=["RAG stands for..."],
        ...     ground_truth=["Retrieval-Augmented Generation..."]
        ... )
        >>>
        >>> print(f"Faithfulness: {results.faithfulness:.2f}")
        >>> print(f"Overall passed: {results.overall_passed}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[EvaluationConfig] = None
    ):
        """Initialize RAG evaluator.

        Args:
            llm: LLM provider for evaluation (judge model)
            embedding_provider: Optional embedding provider for semantic similarity
            config: Evaluation configuration
        """
        self.llm = llm
        self.embedding_provider = embedding_provider
        self.config = config or EvaluationConfig()

        # Initialize calculators
        self.faithfulness_calc = FaithfulnessCalculator(llm, self.config)
        self.context_relevance_calc = ContextRelevanceCalculator(llm)
        self.answer_relevancy_calc = AnswerRelevancyCalculator(llm)
        self.answer_correctness_calc = AnswerCorrectnessCalculator(llm)
        self.hallucination_detector = HallucinationDetector(llm)
        self.context_precision_calc = ContextPrecisionCalculator(llm)
        self.context_recall_calc = ContextRecallCalculator(llm)

    async def evaluate(
        self,
        questions: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: List[str],
        ground_truth: Optional[List[str]] = None,
        relevance_labels: Optional[List[List[float]]] = None
    ) -> EvaluationResult:
        """Evaluate RAG pipeline quality.

        Args:
            questions: List of input questions
            retrieved_contexts: List of context lists (one per question)
            generated_answers: List of generated answers
            ground_truth: Optional list of ground truth answers
            relevance_labels: Optional list of relevance scores per context

        Returns:
            EvaluationResult with all metrics
        """
        import time
        start_time = time.time()

        # Validate inputs
        n_samples = len(questions)
        if len(retrieved_contexts) != n_samples or len(generated_answers) != n_samples:
            raise ValueError("All input lists must have the same length")

        if ground_truth and len(ground_truth) != n_samples:
            raise ValueError("ground_truth must have same length as questions")

        # Evaluate each sample
        sample_results: List[SampleEvaluation] = []

        for i in range(n_samples):
            sample = await self._evaluate_sample(
                question=questions[i],
                contexts=retrieved_contexts[i],
                answer=generated_answers[i],
                ground_truth=ground_truth[i] if ground_truth else None,
                relevance_labels=relevance_labels[i] if relevance_labels else None
            )
            sample_results.append(sample)

            logger.debug(f"Evaluated sample {i+1}/{n_samples}: "
                        f"faithfulness={sample.faithfulness:.2f}")

        # Aggregate results
        result = self._aggregate_results(sample_results, relevance_labels)
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.config = self.config

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    async def _evaluate_sample(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
        relevance_labels: Optional[List[float]] = None
    ) -> SampleEvaluation:
        """Evaluate a single sample."""
        sample = SampleEvaluation(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        # Run evaluations in parallel
        tasks = [
            self.faithfulness_calc.calculate(answer, contexts),
            self.answer_relevancy_calc.calculate(question, answer),
            self.context_precision_calc.calculate(question, contexts),
            self.hallucination_detector.detect(question, answer, contexts),
            self.context_relevance_calc.calculate_batch(question, contexts),
        ]

        # Add ground truth dependent metrics
        if ground_truth:
            tasks.append(self.answer_correctness_calc.calculate(question, answer, ground_truth))
            tasks.append(self.context_recall_calc.calculate(question, contexts, ground_truth))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process faithfulness
        if isinstance(results[0], tuple):
            sample.faithfulness, sample.claims = results[0]
        else:
            logger.warning(f"Faithfulness failed: {results[0]}")

        # Process answer relevancy
        if isinstance(results[1], tuple):
            sample.answer_relevancy, _ = results[1]
        else:
            logger.warning(f"Answer relevancy failed: {results[1]}")

        # Process context precision
        if isinstance(results[2], tuple):
            sample.context_precision, _ = results[2]
        else:
            logger.warning(f"Context precision failed: {results[2]}")

        # Process hallucination
        if isinstance(results[3], tuple):
            sample.hallucination_score, _, _ = results[3]
        else:
            logger.warning(f"Hallucination detection failed: {results[3]}")

        # Process context relevances
        if isinstance(results[4], list):
            sample.context_relevances = results[4]
            # Calculate context relevancy as average
            if sample.context_relevances:
                sample.context_relevancy = sum(
                    cr.relevance_score for cr in sample.context_relevances
                ) / len(sample.context_relevances)
        else:
            logger.warning(f"Context relevance failed: {results[4]}")

        # Process ground truth dependent metrics
        result_idx = 5
        if ground_truth:
            if isinstance(results[result_idx], tuple):
                sample.answer_correctness, _ = results[result_idx]
            result_idx += 1

            if result_idx < len(results) and isinstance(results[result_idx], tuple):
                sample.context_recall, _ = results[result_idx]

        # Determine pass/fail
        sample.failed_metrics = []

        if sample.faithfulness < self.config.faithfulness_threshold:
            sample.failed_metrics.append("faithfulness")
        if sample.answer_relevancy < self.config.answer_relevancy_threshold:
            sample.failed_metrics.append("answer_relevancy")
        if sample.context_precision < self.config.context_precision_threshold:
            sample.failed_metrics.append("context_precision")
        if sample.hallucination_score > self.config.hallucination_threshold:
            sample.failed_metrics.append("hallucination")

        if ground_truth:
            if sample.context_recall < self.config.context_recall_threshold:
                sample.failed_metrics.append("context_recall")

        sample.passed = len(sample.failed_metrics) == 0

        return sample

    def _aggregate_results(
        self,
        samples: List[SampleEvaluation],
        relevance_labels: Optional[List[List[float]]] = None
    ) -> EvaluationResult:
        """Aggregate sample results into overall metrics."""
        n_samples = len(samples)

        if n_samples == 0:
            return EvaluationResult()

        # Calculate averages
        result = EvaluationResult(
            faithfulness=sum(s.faithfulness for s in samples) / n_samples,
            answer_relevancy=sum(s.answer_relevancy for s in samples) / n_samples,
            context_precision=sum(s.context_precision for s in samples) / n_samples,
            context_recall=sum(s.context_recall for s in samples) / n_samples,
            context_relevancy=sum(s.context_relevancy for s in samples) / n_samples,
            answer_correctness=sum(s.answer_correctness for s in samples) / n_samples,
            hallucination_score=sum(s.hallucination_score for s in samples) / n_samples,
            samples=samples,
            total_samples=n_samples,
            passed_samples=sum(1 for s in samples if s.passed),
            failed_samples=sum(1 for s in samples if not s.passed),
        )

        # Calculate ranking metrics if relevance labels provided
        if relevance_labels:
            ndcg_scores = []
            mrr_scores = []

            for labels in relevance_labels:
                if labels:
                    ndcg_scores.append(
                        RankingMetrics.ndcg_at_k(labels, self.config.ndcg_k)
                    )
                    mrr_scores.append(RankingMetrics.mrr(labels))

            if ndcg_scores:
                result.ndcg_at_k = sum(ndcg_scores) / len(ndcg_scores)
            if mrr_scores:
                result.mrr = sum(mrr_scores) / len(mrr_scores)

        # Calculate F1 from context relevance scores
        all_relevances = []
        for sample in samples:
            if sample.context_relevances:
                all_relevances.extend([cr.relevance_score for cr in sample.context_relevances])

        if all_relevances:
            precision = sum(1 for r in all_relevances if r >= 0.5) / len(all_relevances)
            total_relevant = sum(1 for r in all_relevances if r >= 0.5)
            recall = total_relevant / len(all_relevances) if all_relevances else 0
            result.f1_score = RankingMetrics.f1_score(precision, recall)

        # Determine overall pass/fail
        result.overall_passed = (
            result.faithfulness >= self.config.faithfulness_threshold and
            result.answer_relevancy >= self.config.answer_relevancy_threshold and
            result.hallucination_score <= self.config.hallucination_threshold
        )

        return result

    def _generate_recommendations(self, result: EvaluationResult) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []

        if result.faithfulness < self.config.faithfulness_threshold:
            recommendations.append(
                f"LOW FAITHFULNESS ({result.faithfulness:.2f}): "
                "Consider improving context retrieval or adding citation mechanisms "
                "to ground answers in retrieved documents."
            )

        if result.answer_relevancy < self.config.answer_relevancy_threshold:
            recommendations.append(
                f"LOW ANSWER RELEVANCY ({result.answer_relevancy:.2f}): "
                "Improve prompt engineering to ensure answers directly address questions. "
                "Consider few-shot examples."
            )

        if result.context_precision < self.config.context_precision_threshold:
            recommendations.append(
                f"LOW CONTEXT PRECISION ({result.context_precision:.2f}): "
                "Implement reranking to improve context ordering. "
                "Consider cross-encoder models like ms-marco-MiniLM."
            )

        if result.context_recall < self.config.context_recall_threshold:
            recommendations.append(
                f"LOW CONTEXT RECALL ({result.context_recall:.2f}): "
                "Increase retrieval top_k or implement hybrid search (dense + sparse). "
                "Consider query expansion techniques."
            )

        if result.hallucination_score > self.config.hallucination_threshold:
            recommendations.append(
                f"HIGH HALLUCINATION ({result.hallucination_score:.2f}): "
                "Add explicit grounding instructions in prompts. "
                "Implement Self-RAG or CRAG patterns for verification."
            )

        if result.ndcg_at_k < 0.5 and result.ndcg_at_k > 0:
            recommendations.append(
                f"LOW NDCG@{self.config.ndcg_k} ({result.ndcg_at_k:.2f}): "
                "Retrieval ranking needs improvement. "
                "Consider implementing RRF fusion or learned rerankers."
            )

        if not recommendations:
            recommendations.append(
                "All metrics above thresholds. Consider raising thresholds "
                "or evaluating on harder test cases."
            )

        return recommendations

    async def evaluate_single(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> SampleEvaluation:
        """Convenience method to evaluate a single sample.

        Args:
            question: Input question
            contexts: Retrieved context chunks
            answer: Generated answer
            ground_truth: Optional expected answer

        Returns:
            SampleEvaluation with all metrics
        """
        return await self._evaluate_sample(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth
        )


# =============================================================================
# FACTORY AND UTILITIES
# =============================================================================

def create_evaluator(
    llm: LLMProvider,
    config: Optional[EvaluationConfig] = None,
    **kwargs
) -> RAGEvaluator:
    """Factory function to create a RAGEvaluator.

    Args:
        llm: LLM provider for evaluation
        config: Optional configuration
        **kwargs: Additional config overrides

    Returns:
        Configured RAGEvaluator instance
    """
    if config is None:
        config = EvaluationConfig(**kwargs)

    return RAGEvaluator(llm=llm, config=config)


class QuickEvaluator:
    """
    Lightweight evaluator for quick quality checks without full LLM evaluation.

    Uses heuristics and token overlap for fast evaluation.
    Useful for CI/CD pipelines where speed is critical.

    Example:
        >>> quick_eval = QuickEvaluator()
        >>> score = quick_eval.evaluate_faithfulness(answer, contexts)
    """

    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Quick faithfulness check using token overlap."""
        if not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()
        answer_tokens = set(re.findall(r'\b\w+\b', answer.lower()))

        # Filter common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "it", "its", "this", "that", "these", "those",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "as", "into", "through", "during", "before", "after", "above",
                     "below", "between", "under", "again", "further", "then", "once"}

        answer_tokens = answer_tokens - stopwords

        if not answer_tokens:
            return 1.0

        # Count tokens found in context
        found = sum(1 for token in answer_tokens if token in combined_context)
        return found / len(answer_tokens)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """Quick answer relevancy using question term coverage."""
        question_tokens = set(re.findall(r'\b\w+\b', question.lower()))
        answer_lower = answer.lower()

        # Filter stopwords
        important_tokens = {t for t in question_tokens if len(t) > 3}

        if not important_tokens:
            return 0.5

        found = sum(1 for token in important_tokens if token in answer_lower)
        return found / len(important_tokens)

    def evaluate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """Quick context relevancy using term overlap."""
        if not contexts:
            return 0.0

        question_tokens = set(re.findall(r'\b\w+\b', question.lower()))
        important_tokens = {t for t in question_tokens if len(t) > 3}

        if not important_tokens:
            return 0.5

        scores = []
        for context in contexts:
            context_lower = context.lower()
            found = sum(1 for token in important_tokens if token in context_lower)
            scores.append(found / len(important_tokens))

        return sum(scores) / len(scores)

    def evaluate_all(
        self,
        question: str,
        contexts: List[str],
        answer: str
    ) -> Dict[str, float]:
        """Run all quick evaluations."""
        return {
            "faithfulness": self.evaluate_faithfulness(answer, contexts),
            "answer_relevancy": self.evaluate_answer_relevancy(question, answer),
            "context_relevancy": self.evaluate_context_relevancy(question, contexts),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main evaluator
    "RAGEvaluator",
    "QuickEvaluator",
    # Configuration
    "EvaluationConfig",
    # Result types
    "EvaluationResult",
    "SampleEvaluation",
    "MetricResult",
    "Claim",
    "ContextRelevance",
    # Metric calculators
    "FaithfulnessCalculator",
    "ContextRelevanceCalculator",
    "AnswerRelevancyCalculator",
    "AnswerCorrectnessCalculator",
    "HallucinationDetector",
    "ContextPrecisionCalculator",
    "ContextRecallCalculator",
    "ClaimsExtractor",
    "ClaimsVerifier",
    # Ranking metrics
    "RankingMetrics",
    # Prompts
    "EvaluationPrompts",
    # Factory
    "create_evaluator",
    # Protocols
    "LLMProvider",
    "EmbeddingProvider",
]
