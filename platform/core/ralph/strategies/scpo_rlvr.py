"""
V9 Self-Consistency Preference Optimization & RLVR Patterns

Extracted from ralph_loop.py V9 enhancements.
Implements ScPO, RLVR/GRPO, and Multi-Agent Coordination.

Classes:
- ConsistencyPreference: Preference pair for ScPO training
- ScPOState: Self-Consistency Preference Optimization state
- VerifiableReward: Binary reward signal for RLVR
- RLVRState: Reinforcement Learning with Verifiable Rewards state
- AgentMessage: Multi-agent communication message
- AgentCoordinationChannel: Communication channel for coordination
- MultiAgentCoordinationState: Multi-agent collective reasoning state

References:
- ScPO: Self-Consistency Preference Optimization (arxiv 2411.04109, ICML 2025)
- RLVR/GRPO: Reinforcement Learning with Verifiable Rewards (arxiv 2503.06639)
- DeepSeek-R1 training methodology
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConsistencyPreference:
    """
    V9: A preference pair for Self-Consistency Preference Optimization (ScPO).

    Based on arxiv 2411.04109 (ICML 2025):
    Trains consistent answers to be preferred over inconsistent ones.

    Attributes:
        problem_id: Unique identifier for the problem
        consistent_answer: The answer that appeared most frequently
        inconsistent_answer: An inconsistent/minority answer
        consistency_score: Fraction of samples agreeing with consistent_answer
        num_samples: Total samples used
        reasoning_paths: The reasoning chains that led to consistent_answer
        created_at: ISO timestamp of creation
    """
    problem_id: str
    consistent_answer: Any
    inconsistent_answer: Any
    consistency_score: float
    num_samples: int
    reasoning_paths: List[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def preference_strength(self) -> float:
        """How strongly we should prefer the consistent answer (0-1)."""
        return min(1.0, self.consistency_score * 1.2)


@dataclass
class ScPOState:
    """
    V9: Self-Consistency Preference Optimization state.

    ScPO iteratively:
    1. Samples multiple reasoning paths for unsupervised problems
    2. Identifies consistent vs inconsistent answers
    3. Creates preference pairs (prefer consistent over inconsistent)
    4. Trains on these preferences without ground truth labels

    Attributes:
        preference_pairs: Collected preference pairs
        training_iterations: Number of training iterations completed
        consistency_threshold: Minimum consistency to create preference
        num_samples_per_problem: Number of paths to sample
        cumulative_preference_strength: Total preference strength
        problems_evaluated: Number of problems processed
    """
    preference_pairs: List[ConsistencyPreference] = field(default_factory=list)
    training_iterations: int = 0
    consistency_threshold: float = 0.6
    num_samples_per_problem: int = 8
    cumulative_preference_strength: float = 0.0
    problems_evaluated: int = 0

    def add_preference(self, preference: ConsistencyPreference) -> None:
        """Add a new preference pair if it meets the threshold."""
        if preference.consistency_score >= self.consistency_threshold:
            self.preference_pairs.append(preference)
            self.cumulative_preference_strength += preference.preference_strength
            self.problems_evaluated += 1

    def get_training_signal(self) -> float:
        """Get overall training signal strength from preferences."""
        if not self.preference_pairs:
            return 0.0
        return self.cumulative_preference_strength / len(self.preference_pairs)

    def should_update_policy(self) -> bool:
        """Check if we have enough preferences to update policy."""
        return len(self.preference_pairs) >= 4


@dataclass
class VerifiableReward:
    """
    V9: A verifiable reward signal for RLVR/GRPO.

    Based on arxiv 2503.06639 (DeepSeek-R1 training):
    Binary reward for verifiable outcomes (correct/incorrect).

    Attributes:
        sample_id: Unique identifier for this sample
        prompt: The input prompt
        response: The generated response
        is_correct: Binary verification result
        verification_method: How correctness was verified
        reward: 1.0 if correct, 0.0 if not
        confidence: Confidence in verification
    """
    sample_id: str
    prompt: str
    response: str
    is_correct: bool
    verification_method: str  # "exact_match", "code_execution", "math_check", "semantic"
    reward: float = field(init=False)
    confidence: float = 1.0

    def __post_init__(self):
        self.reward = 1.0 if self.is_correct else 0.0


@dataclass
class RLVRState:
    """
    V9: Reinforcement Learning with Verifiable Rewards state.

    GRPO (Group Relative Policy Optimization):
    - Samples multiple outputs per prompt
    - Uses binary verifiable rewards
    - KL-regularized contrastive loss
    - Amplifies probability of success

    Attributes:
        samples: Collected verified samples
        group_size: Samples per prompt for GRPO
        kl_coefficient: KL regularization strength
        reference_policy_divergence: Track drift from reference
        success_rate: Running success rate
        policy_updates: Number of policy updates performed
        mean_reward: Running mean reward
        reward_variance: Running reward variance
        contrastive_pairs_created: Total contrastive pairs created
    """
    samples: List[VerifiableReward] = field(default_factory=list)
    group_size: int = 4
    kl_coefficient: float = 0.1
    reference_policy_divergence: float = 0.0
    success_rate: float = 0.0
    policy_updates: int = 0
    mean_reward: float = 0.0
    reward_variance: float = 0.0
    contrastive_pairs_created: int = 0

    def add_sample(self, sample: VerifiableReward) -> None:
        """Add a new verified sample."""
        self.samples.append(sample)
        n = len(self.samples)
        old_mean = self.mean_reward
        self.mean_reward += (sample.reward - self.mean_reward) / n
        self.reward_variance += (sample.reward - old_mean) * (sample.reward - self.mean_reward)
        if sample.is_correct:
            self.success_rate = (self.success_rate * (n - 1) + 1.0) / n
        else:
            self.success_rate = (self.success_rate * (n - 1)) / n

    def get_normalized_reward(self, reward: float) -> float:
        """
        GRPO reward normalization: (r - mean) / std.

        This creates the contrastive signal: positive for above-average,
        negative for below-average responses.
        """
        if len(self.samples) < 2:
            return reward
        std = math.sqrt(self.reward_variance / len(self.samples)) if self.reward_variance > 0 else 1.0
        return (reward - self.mean_reward) / (std + 1e-8)

    def create_contrastive_pairs(self) -> List[Tuple[VerifiableReward, VerifiableReward]]:
        """Create contrastive pairs (positive, negative) for training."""
        positives = [s for s in self.samples if s.is_correct]
        negatives = [s for s in self.samples if not s.is_correct]

        pairs = []
        for pos in positives:
            for neg in negatives:
                pairs.append((pos, neg))
                self.contrastive_pairs_created += 1

        return pairs


@dataclass
class AgentMessage:
    """
    V9: Message for multi-agent communication.

    Attributes:
        sender_id: ID of the sending agent
        receiver_id: ID of receiver ("*" for broadcast)
        message_type: Type of message
        content: Message content
        priority: Priority level (0-1)
        timestamp: ISO timestamp
        requires_response: Whether response is expected
        in_response_to: ID of message being responded to
    """
    sender_id: str
    receiver_id: str  # "*" for broadcast
    message_type: str  # "proposal", "critique", "agreement", "request", "response"
    content: str
    priority: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    requires_response: bool = False
    in_response_to: Optional[str] = None


@dataclass
class AgentCoordinationChannel:
    """
    V9: Communication channel for multi-agent coordination.

    Enables structured communication between different agent perspectives
    (from V8 self-play) for collective reasoning.

    Attributes:
        channel_id: Unique channel identifier
        participants: List of agent IDs in channel
        messages: Message history
        consensus_reached: Whether consensus was achieved
        consensus_content: The consensus decision content
    """
    channel_id: str
    participants: List[str]
    messages: List[AgentMessage] = field(default_factory=list)
    consensus_reached: bool = False
    consensus_content: str = ""

    def broadcast(self, sender_id: str, content: str, msg_type: str = "proposal") -> None:
        """Broadcast message to all participants."""
        msg = AgentMessage(
            sender_id=sender_id,
            receiver_id="*",
            message_type=msg_type,
            content=content
        )
        self.messages.append(msg)

    def send_direct(self, sender_id: str, receiver_id: str, content: str,
                    msg_type: str = "response") -> None:
        """Send direct message to specific agent."""
        msg = AgentMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=msg_type,
            content=content
        )
        self.messages.append(msg)

    def get_messages_for(self, agent_id: str) -> List[AgentMessage]:
        """Get all messages relevant to an agent (direct + broadcast)."""
        return [m for m in self.messages
                if m.receiver_id == agent_id or m.receiver_id == "*"]

    def propose_consensus(self, content: str) -> None:
        """Propose a consensus decision."""
        self.consensus_content = content

    def finalize_consensus(self) -> str:
        """Finalize consensus after agreement."""
        self.consensus_reached = True
        return self.consensus_content


@dataclass
class MultiAgentCoordinationState:
    """
    V9: Multi-Agent Coordination state for collective reasoning.

    Extends V8 self-play with structured communication protocols:
    - Channels for different discussion topics
    - Consensus mechanisms for collective decisions
    - Role-based coordination (leader election, task allocation)

    Attributes:
        channels: Active coordination channels
        active_agents: List of active agent IDs
        coordinator_agent: Current leader/coordinator
        coordination_rounds: Number of coordination rounds
        consensus_history: History of consensus decisions
        messages_exchanged: Total messages exchanged
        consensus_attempts: Total consensus attempts
        successful_consensus: Number of successful consensus
    """
    channels: Dict[str, AgentCoordinationChannel] = field(default_factory=dict)
    active_agents: List[str] = field(default_factory=list)
    coordinator_agent: Optional[str] = None
    coordination_rounds: int = 0
    consensus_history: List[Dict[str, Any]] = field(default_factory=list)
    messages_exchanged: int = 0
    consensus_attempts: int = 0
    successful_consensus: int = 0

    def create_channel(self, channel_id: str, participants: List[str]) -> AgentCoordinationChannel:
        """Create a new coordination channel."""
        channel = AgentCoordinationChannel(
            channel_id=channel_id,
            participants=participants
        )
        self.channels[channel_id] = channel
        return channel

    def elect_coordinator(self, agent_scores: Dict[str, float]) -> str:
        """Elect coordinator based on agent performance scores."""
        if not agent_scores:
            return ""
        self.coordinator_agent = max(agent_scores, key=lambda x: agent_scores.get(x, 0.0))
        return self.coordinator_agent

    def run_coordination_round(self, topic: str) -> Dict[str, Any]:
        """Run a coordination round on a topic."""
        self.coordination_rounds += 1

        channel_id = f"round_{self.coordination_rounds}_{topic[:20]}"
        if channel_id not in self.channels:
            self.create_channel(channel_id, self.active_agents)

        channel = self.channels[channel_id]

        result = {
            "round": self.coordination_rounds,
            "topic": topic,
            "channel_id": channel_id,
            "participants": channel.participants,
            "messages_count": len(channel.messages),
            "consensus_reached": channel.consensus_reached,
            "consensus": channel.consensus_content if channel.consensus_reached else None
        }

        if channel.consensus_reached:
            self.successful_consensus += 1
            self.consensus_history.append(result)

        self.consensus_attempts += 1
        return result

    def get_coordination_effectiveness(self) -> float:
        """Calculate coordination effectiveness (0-1)."""
        if self.consensus_attempts == 0:
            return 0.0
        return self.successful_consensus / self.consensus_attempts


__all__ = [
    "ConsistencyPreference",
    "ScPOState",
    "VerifiableReward",
    "RLVRState",
    "AgentMessage",
    "AgentCoordinationChannel",
    "MultiAgentCoordinationState",
]
