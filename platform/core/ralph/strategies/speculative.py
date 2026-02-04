"""
V11 Speculative Decoding, Adaptive RAG & Reward Safety

Extracted from ralph_loop.py V11 enhancements.
Implements parallel hypothesis generation, efficient reasoning, adaptive retrieval,
and reward hacking detection.

Classes:
- SpeculativeHypothesis: A hypothesis in speculative parallel execution
- SpeculativeDecodingState: State for speculative parallel hypothesis generation
- DraftStep: A concise draft step in Chain-of-Draft reasoning
- ChainOfDraftState: State for Chain-of-Draft efficient reasoning
- RetrievalDecision: A decision about whether to retrieve external knowledge
- AdaptiveRAGState: Adaptive Retrieval-Augmented Generation state
- RewardHackingSignal: A detected reward hacking signal
- RewardHackingDetectorState: State for reward hacking detection
- MetaJudgment: A meta-judgment (judgment of a judgment)
- MetaRewardState: Meta-Reward Model state
- CausalIntervention: A causal intervention for improvement attribution
- ImprovementAttributionState: State for causal attribution of improvements

References:
- PEARL: Adaptive draft length (ICLR 2025)
- Chain-of-Draft: 92.4% token compression (arxiv 2502.18600)
- Adaptive RAG: INKER, DynamicRAG (ICLR 2026)
- A2RM: Adversarial-Augmented Reward Model (ICLR 2026)
- Meta-Rewarding Language Models (arxiv 2407.19594)
- Causal Head Gating (CHG) and interchange intervention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SpeculativeHypothesis:
    """
    V11: A hypothesis in speculative parallel execution.

    Based on PEARL (ICLR 2025) and Speculative Speculative Decoding (SSD).
    Multiple hypotheses are generated in parallel and verified asynchronously.

    Attributes:
        hypothesis_id: Unique identifier
        content: The hypothesis content
        confidence: Prior probability estimate
        generation_cost: Tokens used to generate
        verification_status: pending, verified, or rejected
        verification_result: Boolean result of verification
        verification_reasoning: Explanation of verification
        verification_cost: Tokens used for verification
        timestamp: Unix timestamp of creation
        created_at: ISO timestamp of creation
    """
    hypothesis_id: int
    content: str
    confidence: float
    generation_cost: int
    verification_status: str = "pending"
    verification_result: Optional[bool] = None
    verification_reasoning: str = ""
    verification_cost: int = 0
    timestamp: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_verified(self) -> bool:
        return self.verification_status == "verified"


@dataclass
class SpeculativeDecodingState:
    """
    V11: State for speculative parallel hypothesis generation.

    Key concepts from research:
    - PEARL: Adaptive draft length to minimize waiting
    - SSD: Speculate on anticipated verification outcomes
    - Query-and-Correct: Decouple drafting from verification

    Attributes:
        hypotheses: List of generated hypotheses
        verified_count: Number of verified hypotheses
        rejected_count: Number of rejected hypotheses
        total_speculation_tokens: Tokens spent on speculation
        total_verification_tokens: Tokens spent on verification
        optimal_batch_size: Learned optimal batch size
        acceptance_rate: Historical acceptance rate
        speculation_depth: How far ahead to speculate
        total_hypotheses_generated: Total generated across batches
        total_hypotheses_accepted: Total accepted hypotheses
        speedup_factor: Current speedup vs sequential
    """
    hypotheses: List[SpeculativeHypothesis] = field(default_factory=list)
    verified_count: int = 0
    rejected_count: int = 0
    total_speculation_tokens: int = 0
    total_verification_tokens: int = 0
    optimal_batch_size: int = 4
    acceptance_rate: float = 0.5
    speculation_depth: int = 3
    total_hypotheses_generated: int = 0
    total_hypotheses_accepted: int = 0
    speedup_factor: float = 1.0

    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on acceptance rate."""
        if self.acceptance_rate > 0.8:
            return min(8, self.optimal_batch_size + 1)
        elif self.acceptance_rate < 0.3:
            return max(2, self.optimal_batch_size - 1)
        return self.optimal_batch_size

    def add_hypothesis(self, hypothesis: SpeculativeHypothesis) -> None:
        """Add an existing hypothesis to tracking."""
        self.hypotheses.append(hypothesis)
        if len(self.hypotheses) > 50:
            self.hypotheses = self.hypotheses[-50:]

    def add_new_hypothesis(self, content: str, confidence: float, cost: int) -> SpeculativeHypothesis:
        """Add a new speculative hypothesis."""
        h = SpeculativeHypothesis(
            hypothesis_id=len(self.hypotheses),
            content=content,
            confidence=confidence,
            generation_cost=cost
        )
        self.hypotheses.append(h)
        self.total_speculation_tokens += cost
        return h

    def verify_hypothesis(self, hypothesis_id: int, accepted: bool, reasoning: str = "") -> None:
        """Record verification result for a hypothesis."""
        if 0 <= hypothesis_id < len(self.hypotheses):
            h = self.hypotheses[hypothesis_id]
            h.verification_status = "verified" if accepted else "rejected"
            h.verification_result = accepted
            h.verification_reasoning = reasoning
            if accepted:
                self.verified_count += 1
            else:
                self.rejected_count += 1

    def get_acceptance_rate(self) -> float:
        """Calculate current acceptance rate."""
        total = self.verified_count + self.rejected_count
        if total == 0:
            return self.acceptance_rate
        return self.verified_count / total

    def get_speculation_efficiency(self) -> float:
        """Tokens saved per accepted hypothesis."""
        if self.verified_count == 0:
            return 0.0
        return self.total_speculation_tokens / max(1, self.verified_count)


@dataclass
class DraftStep:
    """
    V11: A concise draft step in Chain-of-Draft reasoning.

    Based on arxiv 2502.18600 - "Thinking Faster by Writing Less".
    Draft steps capture only essential information, achieving
    92.4% fewer tokens than Chain-of-Thought while maintaining accuracy.

    Attributes:
        step_index: Position in the draft chain
        draft_content: Minimal, information-dense step
        token_count: Number of tokens in this step
        captures_key_insight: Whether step captures key insight
        is_verified: Whether step has been verified/expanded
        expansion_available: Whether full expansion can be generated
        created_at: ISO timestamp of creation
    """
    step_index: int
    draft_content: str
    token_count: int
    captures_key_insight: bool = True
    is_verified: bool = False
    expansion_available: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ChainOfDraftState:
    """
    V11: State for Chain-of-Draft efficient reasoning.

    Humans draft concise notes; LLMs should too.
    CoD produces minimal intermediate thoughts while preserving accuracy.

    Attributes:
        draft_chains: List of completed draft chains
        total_draft_tokens: Total tokens in drafts
        total_equivalent_cot_tokens: Estimated CoT equivalent
        compression_ratio: Target draft/cot ratio
        total_chains: Number of chains created
        average_steps_per_chain: Running average steps
    """
    draft_chains: List[List[DraftStep]] = field(default_factory=list)
    total_draft_tokens: int = 0
    total_equivalent_cot_tokens: int = 0
    compression_ratio: float = 0.1
    total_chains: int = 0
    average_steps_per_chain: float = 0.0

    def add_draft_chain(self, steps: List[DraftStep]) -> None:
        """Add a completed draft chain."""
        self.draft_chains.append(steps)
        chain_tokens = sum(s.token_count for s in steps)
        self.total_draft_tokens += chain_tokens
        self.total_equivalent_cot_tokens += int(chain_tokens / self.compression_ratio)
        self.total_chains += 1
        total_steps = sum(len(c) for c in self.draft_chains)
        self.average_steps_per_chain = total_steps / self.total_chains
        if len(self.draft_chains) > 20:
            self.draft_chains = self.draft_chains[-20:]

    def get_token_savings(self) -> int:
        """Calculate tokens saved vs traditional CoT."""
        return max(0, self.total_equivalent_cot_tokens - self.total_draft_tokens)

    def get_efficiency_ratio(self) -> float:
        """Get actual compression efficiency."""
        if self.total_equivalent_cot_tokens == 0:
            return 1.0
        return self.total_draft_tokens / self.total_equivalent_cot_tokens


@dataclass
class RetrievalDecision:
    """
    V11: A decision about whether to retrieve external knowledge.

    Based on Adaptive RAG research (INKER, DynamicRAG, ICLR 2026).

    Attributes:
        query: The query being evaluated
        should_retrieve: Whether to retrieve
        confidence: Confidence in decision
        reasoning: Explanation of decision
        retrieval_type: Type of retrieval
        retrieved_context: Retrieved content
        retrieval_latency_ms: Retrieval latency
        context_relevance_score: Relevance of retrieved content
        novelty_score: Query novelty estimate
        retrieval_result: Retrieved content
        was_helpful: Whether retrieval helped
    """
    query: str
    should_retrieve: bool
    confidence: float
    reasoning: str = ""
    retrieval_type: str = "none"
    retrieved_context: str = ""
    retrieval_latency_ms: float = 0.0
    context_relevance_score: float = 0.0
    novelty_score: float = 0.0
    retrieval_result: Optional[str] = None
    was_helpful: Optional[bool] = None


@dataclass
class AdaptiveRAGState:
    """
    V11: Adaptive Retrieval-Augmented Generation state.

    Dynamically decides whether to retrieve based on:
    - Model's internal knowledge confidence
    - Query complexity and novelty
    - Past retrieval effectiveness

    Attributes:
        retrieval_decisions: List of retrieval decisions
        total_retrievals: Total retrievals performed
        successful_retrievals: Retrievals that helped
        confidence_threshold: Below this, consider retrieval
        novelty_threshold: Above this, retrieve for novel queries
        internal_knowledge_hits: Internal knowledge hits
        external_knowledge_hits: External knowledge hits
        total_decisions: Total decisions made
        retrieval_count: Decisions to retrieve
        skip_count: Decisions to skip
        retrieval_success_rate: Running success rate
    """
    retrieval_decisions: List[RetrievalDecision] = field(default_factory=list)
    total_retrievals: int = 0
    successful_retrievals: int = 0
    confidence_threshold: float = 0.7
    novelty_threshold: float = 0.5
    internal_knowledge_hits: int = 0
    external_knowledge_hits: int = 0
    total_decisions: int = 0
    retrieval_count: int = 0
    skip_count: int = 0
    retrieval_success_rate: float = 0.0

    def record_decision(self, decision: RetrievalDecision) -> None:
        """Record a retrieval decision."""
        self.retrieval_decisions.append(decision)
        self.total_decisions += 1
        if decision.should_retrieve:
            self.total_retrievals += 1
            self.retrieval_count += 1
            if decision.retrieval_type == "internal":
                self.internal_knowledge_hits += 1
            else:
                self.external_knowledge_hits += 1
        else:
            self.skip_count += 1
        if len(self.retrieval_decisions) > 50:
            self.retrieval_decisions = self.retrieval_decisions[-50:]
        self.retrieval_success_rate = self.get_retrieval_effectiveness()

    def record_retrieval_outcome(self, helpful: bool) -> None:
        """Record whether retrieval was helpful."""
        if helpful:
            self.successful_retrievals += 1
        self.retrieval_success_rate = self.get_retrieval_effectiveness()

    def get_retrieval_effectiveness(self) -> float:
        """Get retrieval success rate."""
        if self.total_retrievals == 0:
            return 0.0
        return self.successful_retrievals / self.total_retrievals

    def should_retrieve(self, confidence: float, novelty: float) -> bool:
        """Decide whether to retrieve based on confidence and novelty."""
        if confidence < self.confidence_threshold:
            return True
        if novelty > self.novelty_threshold:
            return True
        if self.get_retrieval_effectiveness() > 0.7:
            return True
        return False


@dataclass
class RewardHackingSignal:
    """
    V11: A detected reward hacking signal.

    Based on Anthropic's "Natural Emergent Misalignment" and
    A2RM (Adversarial-Augmented Reward Model, ICLR 2026).

    Attributes:
        signal_type: Type of hacking detected
        severity: Severity level (0-1)
        detection_method: How it was detected
        description: Description of the signal
        signal_id: Unique identifier
        affected_metric: Which metric was affected
        timestamp: Unix timestamp
        mitigation_applied: What mitigation was applied
        mitigation_action: The mitigation action taken
        detected_at: ISO timestamp of detection
    """
    signal_type: str  # proxy_gaming, specification_gaming, reward_tampering
    severity: float
    detection_method: str
    description: str = ""
    signal_id: int = 0
    affected_metric: str = "fitness"
    timestamp: float = 0.0
    mitigation_applied: Optional[str] = None
    mitigation_action: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RewardHackingDetectorState:
    """
    V11: State for reward hacking detection and mitigation.

    Implements patterns from:
    - Evaluator Stress Tests (arxiv 2507.05619)
    - A2RM: Adversarial-Augmented Reward Model
    - APRM: Adversarial Training for PRMs

    Attributes:
        detected_signals: List of detected signals
        stress_tests_run: Number of stress tests run
        vulnerabilities_found: Number of vulnerabilities found
        vulnerabilities_patched: Number patched
        proxy_divergence_threshold: Threshold for proxy divergence
        suspicious_improvement_threshold: Threshold for suspicious improvements
        reward_history: True reward history
        proxy_reward_history: Proxy reward history
        total_checks: Total reward checks
        total_detections: Total signals detected
        mitigation_actions_taken: Total mitigations applied
    """
    detected_signals: List[RewardHackingSignal] = field(default_factory=list)
    stress_tests_run: int = 0
    vulnerabilities_found: int = 0
    vulnerabilities_patched: int = 0
    proxy_divergence_threshold: float = 0.3
    suspicious_improvement_threshold: float = 0.5
    reward_history: List[float] = field(default_factory=list)
    proxy_reward_history: List[float] = field(default_factory=list)
    total_checks: int = 0
    total_detections: int = 0
    mitigation_actions_taken: int = 0

    def add_signal(self, signal: RewardHackingSignal) -> None:
        """Record a detected reward hacking signal."""
        signal.signal_id = len(self.detected_signals)
        self.detected_signals.append(signal)
        self.vulnerabilities_found += 1
        if len(self.detected_signals) > 50:
            self.detected_signals = self.detected_signals[-50:]

    def create_signal(self, signal_type: str, description: str, severity: float,
                      affected_metric: str, detection_method: str) -> RewardHackingSignal:
        """Create and record a reward hacking signal."""
        signal = RewardHackingSignal(
            signal_type=signal_type,
            severity=severity,
            detection_method=detection_method,
            description=description,
            signal_id=len(self.detected_signals),
            affected_metric=affected_metric
        )
        self.add_signal(signal)
        return signal

    def check_proxy_divergence(self, true_reward: float, proxy_reward: float) -> bool:
        """Check if proxy reward is diverging from true reward."""
        self.reward_history.append(true_reward)
        self.proxy_reward_history.append(proxy_reward)

        if len(self.reward_history) < 5:
            return False

        recent_true = self.reward_history[-5:]
        recent_proxy = self.proxy_reward_history[-5:]
        divergence = sum(abs(t - p) for t, p in zip(recent_true, recent_proxy)) / 5

        return divergence > self.proxy_divergence_threshold

    def check_suspicious_improvement(self, improvement: float) -> bool:
        """Check for suspiciously large improvements."""
        return improvement > self.suspicious_improvement_threshold

    def apply_mitigation(self, signal_id: int, action: str) -> None:
        """Apply mitigation for a detected signal."""
        if 0 <= signal_id < len(self.detected_signals):
            self.detected_signals[signal_id].mitigation_applied = action
            self.detected_signals[signal_id].mitigation_action = action
            self.mitigation_actions_taken += 1
            self.vulnerabilities_patched += 1


@dataclass
class MetaJudgment:
    """
    V11: A meta-judgment (judgment of a judgment).

    Based on "Meta-Rewarding Language Models" (arxiv 2407.19594).
    The model judges its own judgments to improve judgment skills.

    Attributes:
        original_judgment: The original judgment
        meta_judgment: Judgment of the judgment
        meta_score: Quality of original judgment
        judgment_id: Unique identifier
        original_score: Original judgment score
        judgment_type: Type of judgment
        confidence: Confidence in meta-judgment
        reasoning: Reasoning for meta-judgment
        improvement_suggestion: Suggested improvement
        created_at: ISO timestamp of creation
    """
    original_judgment: str
    meta_judgment: str
    meta_score: float
    judgment_id: int = 0
    original_score: float = 0.0
    judgment_type: str = "reward"
    confidence: float = 0.5
    reasoning: Optional[str] = None
    improvement_suggestion: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MetaRewardState:
    """
    V11: Meta-Reward Model state.

    Implements LLM-as-a-Meta-Judge pattern:
    - Model judges responses (level 1)
    - Model judges its judgments (level 2 - meta)
    - Feedback improves both response and judgment quality

    Attributes:
        meta_judgments: List of meta-judgments
        judgment_improvement_rate: Rate of judgment improvement
        response_improvement_rate: Rate of response improvement
        level1_updates: Response improvements
        level2_updates: Judgment improvements
        total_judgments: Total judgments made
        average_meta_score: Running average meta score
        judgment_consistency: Consistency of meta-judgments
    """
    meta_judgments: List[MetaJudgment] = field(default_factory=list)
    judgment_improvement_rate: float = 0.0
    response_improvement_rate: float = 0.0
    level1_updates: int = 0
    level2_updates: int = 0
    total_judgments: int = 0
    average_meta_score: float = 0.0
    judgment_consistency: float = 0.0

    def add_meta_judgment(self, judgment: MetaJudgment) -> None:
        """Add a meta-judgment."""
        judgment.judgment_id = len(self.meta_judgments)
        self.meta_judgments.append(judgment)
        self.total_judgments += 1
        self.average_meta_score = (
            (self.average_meta_score * (self.total_judgments - 1) + judgment.meta_score)
            / self.total_judgments
        )
        if self.total_judgments > 1:
            scores = [mj.meta_score for mj in self.meta_judgments[-10:]]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            self.judgment_consistency = max(0, 1 - variance)
        if len(self.meta_judgments) > 50:
            self.meta_judgments = self.meta_judgments[-50:]

    def create_meta_judgment(self, original_judgment: str, original_score: float,
                             meta_judgment: str, meta_score: float, suggestion: str) -> MetaJudgment:
        """Create and add a meta-judgment."""
        mj = MetaJudgment(
            original_judgment=original_judgment,
            meta_judgment=meta_judgment,
            meta_score=meta_score,
            judgment_id=len(self.meta_judgments),
            original_score=original_score,
            improvement_suggestion=suggestion
        )
        self.add_meta_judgment(mj)
        return mj

    def get_meta_judgment_quality(self) -> float:
        """Average quality of meta-judgments."""
        if not self.meta_judgments:
            return 0.0
        return sum(mj.meta_score for mj in self.meta_judgments) / len(self.meta_judgments)

    def update_improvement_rates(self, response_improved: bool, judgment_improved: bool) -> None:
        """Update improvement tracking."""
        alpha = 0.1
        if response_improved:
            self.level1_updates += 1
            self.response_improvement_rate = alpha * 1.0 + (1 - alpha) * self.response_improvement_rate
        else:
            self.response_improvement_rate = alpha * 0.0 + (1 - alpha) * self.response_improvement_rate

        if judgment_improved:
            self.level2_updates += 1
            self.judgment_improvement_rate = alpha * 1.0 + (1 - alpha) * self.judgment_improvement_rate
        else:
            self.judgment_improvement_rate = alpha * 0.0 + (1 - alpha) * self.judgment_improvement_rate


@dataclass
class CausalIntervention:
    """
    V11: A causal intervention for improvement attribution.

    Based on Causal Head Gating (CHG) and interchange intervention methods.

    Attributes:
        intervention_type: Type of intervention
        target_component: What was intervened on
        causal_effect: Difference in performance
        intervention_id: Unique identifier
        baseline_performance: Performance before intervention
        intervened_performance: Performance after intervention
        baseline_value: Alias for baseline_performance
        intervened_value: Alias for intervened_performance
        confidence: Confidence in the result
        timestamp: Unix timestamp
        interpretation: Human-readable interpretation
    """
    intervention_type: str
    target_component: str
    causal_effect: float
    intervention_id: int = 0
    baseline_performance: float = 0.0
    intervened_performance: float = 0.0
    baseline_value: float = 0.0
    intervened_value: float = 0.0
    confidence: float = 0.5
    timestamp: float = 0.0
    interpretation: str = ""


@dataclass
class ImprovementAttributionState:
    """
    V11: State for causal attribution of improvements.

    Tracks which changes actually caused improvements using
    causal intervention analysis.

    Attributes:
        interventions: List of interventions
        attributed_improvements: Component to causal effect mapping
        counterfactual_tests: Number of counterfactual tests
        significant_attributions: Attributions with high confidence
        total_interventions: Total interventions performed
        attribution_confidence: Running confidence score
    """
    interventions: List[CausalIntervention] = field(default_factory=list)
    attributed_improvements: Dict[str, float] = field(default_factory=dict)
    counterfactual_tests: int = 0
    significant_attributions: int = 0
    total_interventions: int = 0
    attribution_confidence: float = 0.0

    def add_intervention(self, intervention: CausalIntervention) -> None:
        """Add a causal intervention result."""
        intervention.intervention_id = len(self.interventions)
        self.interventions.append(intervention)
        self.counterfactual_tests += 1
        self.total_interventions += 1

        target = intervention.target_component
        effect = intervention.causal_effect
        if target in self.attributed_improvements:
            self.attributed_improvements[target] = (
                self.attributed_improvements[target] + effect
            ) / 2
        else:
            self.attributed_improvements[target] = effect

        if intervention.confidence > 0.7:
            self.significant_attributions += 1

        self.attribution_confidence = self._calculate_attribution_confidence()

        if len(self.interventions) > 50:
            self.interventions = self.interventions[-50:]

    def _calculate_attribution_confidence(self) -> float:
        """Calculate overall attribution confidence."""
        if not self.interventions:
            return 0.0
        return sum(i.confidence for i in self.interventions) / len(self.interventions)

    def create_intervention(self, intervention_type: str, target: str,
                            baseline: float, intervened: float, interpretation: str) -> CausalIntervention:
        """Create and add a causal intervention result."""
        effect = baseline - intervened
        confidence = min(1.0, abs(effect) / max(0.01, baseline))

        intervention = CausalIntervention(
            intervention_type=intervention_type,
            target_component=target,
            causal_effect=effect,
            intervention_id=len(self.interventions),
            baseline_performance=baseline,
            intervened_performance=intervened,
            baseline_value=baseline,
            intervened_value=intervened,
            confidence=confidence,
            interpretation=interpretation
        )
        self.add_intervention(intervention)
        return intervention

    def get_top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N components by causal contribution."""
        sorted_attrs = sorted(
            self.attributed_improvements.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_attrs[:n]


__all__ = [
    "SpeculativeHypothesis",
    "SpeculativeDecodingState",
    "DraftStep",
    "ChainOfDraftState",
    "RetrievalDecision",
    "AdaptiveRAGState",
    "RewardHackingSignal",
    "RewardHackingDetectorState",
    "MetaJudgment",
    "MetaRewardState",
    "CausalIntervention",
    "ImprovementAttributionState",
]
