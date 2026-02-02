#!/usr/bin/env python3
"""
Claude Flow V3 - Enhanced Multi-Agent Orchestration with Swarm Patterns

V3 Enhancements:
1. Dynamic swarm spawning based on task complexity
2. Consensus-based decision making (multiple agents vote)
3. Letta memory integration for cross-agent context
4. Evaluator-Optimizer loops (self-critique)
5. Circuit breaker for resilience
6. Parallel agent execution with asyncio.gather()

Usage:
    from core.orchestration.claude_flow_v3 import ClaudeFlowV3, SwarmConfig

    # Create enhanced flow
    flow = ClaudeFlowV3.create_research_swarm()
    result = await flow.run_with_consensus("Analyze market trends", consensus_threshold=0.7)
"""

from __future__ import annotations

import asyncio
import os
import json
import time
from datetime import datetime
from typing import Any, Optional, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import structlog
from pydantic import BaseModel, Field

# Import base classes from claude_flow
from core.orchestration.claude_flow import (
    AgentRole,
    MessageType,
    FlowMessage,
    AgentConfig,
    FlowConfig,
    AgentResult,
    FlowResult,
    ClaudeAgent,
    ClaudeFlow,
)

logger = structlog.get_logger(__name__)

T = TypeVar('T')


# ========== V3 SWARM PATTERNS ==========

class SwarmStrategy(str, Enum):
    """Strategies for swarm coordination."""
    SEQUENTIAL = "sequential"           # Planner → Executor → Reviewer
    PARALLEL = "parallel"               # All agents work simultaneously
    HIERARCHICAL = "hierarchical"       # Coordinator delegates to sub-agents
    CONSENSUS = "consensus"             # Multiple agents vote on output
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"  # Generate → Evaluate → Iterate


class ConsensusMethod(str, Enum):
    """Methods for reaching consensus."""
    MAJORITY = "majority"               # 51%+ agreement
    SUPERMAJORITY = "supermajority"     # 67%+ agreement
    UNANIMOUS = "unanimous"             # 100% agreement
    WEIGHTED = "weighted"               # Weighted by agent confidence


@dataclass
class ConsensusVote:
    """A vote from an agent in consensus mode."""
    agent_name: str
    decision: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Result of a consensus process."""
    reached: bool
    winning_decision: Optional[str]
    vote_count: dict[str, int] = field(default_factory=dict)
    total_confidence: float = 0.0
    votes: list[ConsensusVote] = field(default_factory=list)
    method: ConsensusMethod = ConsensusMethod.MAJORITY


class SwarmConfig(BaseModel):
    """Configuration for V3 swarm orchestration."""
    strategy: SwarmStrategy = SwarmStrategy.SEQUENTIAL
    consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY
    consensus_threshold: float = 0.5
    max_swarm_size: int = 8
    min_swarm_size: int = 2
    enable_letta_memory: bool = True
    letta_agent_id: Optional[str] = None  # Letta Cloud agent for shared memory
    evaluator_max_iterations: int = 3
    circuit_breaker_threshold: int = 3  # Failures before opening


# ========== CIRCUIT BREAKER ==========

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking calls
    HALF_OPEN = "half_open" # Testing recovery


class GOAPRecoveryAction(str, Enum):
    """V40: GOAP-style recovery actions when circuit breaker opens."""
    HARD_RESET = "hard_reset"       # Reinject original goal into context
    GOAL_REMINDER = "goal_reminder" # Add goal reminder to next prompt
    SLOW_DOWN = "slow_down"         # Reduce action frequency, add verification
    RE_ANCHOR = "re_anchor"         # Re-anchor to goal with alignment check
    FALLBACK_AGENT = "fallback_agent"  # Switch to fallback agent
    REDUCE_SCOPE = "reduce_scope"   # Reduce task scope for recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for agent resilience."""
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure_time: Optional[float] = None

    def record_success(self) -> None:
        """Record a successful call."""
        self.failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                failures=self.failures,
                threshold=self.failure_threshold
            )

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        return True  # HALF_OPEN allows one attempt

    def get_recovery_action(self) -> GOAPRecoveryAction:
        """V40: Get recommended GOAP-style recovery action based on failure pattern.

        This uses A*-style heuristics to determine the best recovery action:
        - 1-2 failures: goal_reminder (low cost, maintain momentum)
        - 3 failures: slow_down (moderate cost, add verification)
        - 4+ failures: hard_reset (high cost, reinject goal)
        - If in HALF_OPEN and fails again: fallback_agent
        """
        if self.state == CircuitState.HALF_OPEN:
            # Already tried recovery, use fallback
            return GOAPRecoveryAction.FALLBACK_AGENT

        if self.failures >= 4:
            return GOAPRecoveryAction.HARD_RESET
        elif self.failures >= 3:
            return GOAPRecoveryAction.SLOW_DOWN
        elif self.failures >= 2:
            return GOAPRecoveryAction.RE_ANCHOR
        else:
            return GOAPRecoveryAction.GOAL_REMINDER

    def apply_recovery(self, action: GOAPRecoveryAction, context: dict[str, Any]) -> dict[str, Any]:
        """V40: Apply GOAP recovery action to context.

        Args:
            action: The recovery action to apply
            context: Current execution context

        Returns:
            Modified context with recovery adjustments
        """
        recovery_context = context.copy()

        if action == GOAPRecoveryAction.HARD_RESET:
            # Reinject original goal into context
            recovery_context["_recovery_mode"] = "hard_reset"
            recovery_context["_recovery_prompt"] = (
                "CRITICAL RECOVERY: Previous attempts failed. "
                "Returning to original goal. Focus on core objective only."
            )
            recovery_context["_action_frequency"] = "slow"
            logger.warning("goap_recovery_hard_reset", failures=self.failures)

        elif action == GOAPRecoveryAction.GOAL_REMINDER:
            # Add goal reminder
            recovery_context["_recovery_mode"] = "goal_reminder"
            recovery_context["_recovery_prompt"] = (
                "Reminder: Stay focused on the stated goal. "
                "Verify each step aligns with the objective."
            )
            logger.info("goap_recovery_goal_reminder", failures=self.failures)

        elif action == GOAPRecoveryAction.SLOW_DOWN:
            # Reduce action frequency, add verification
            recovery_context["_recovery_mode"] = "slow_down"
            recovery_context["_action_frequency"] = "slow"
            recovery_context["_verification_required"] = True
            recovery_context["_recovery_prompt"] = (
                "Slowing down execution. Verify each action before proceeding. "
                "Check alignment with goal at each step."
            )
            logger.info("goap_recovery_slow_down", failures=self.failures)

        elif action == GOAPRecoveryAction.RE_ANCHOR:
            # Re-anchor to goal with alignment check
            recovery_context["_recovery_mode"] = "re_anchor"
            recovery_context["_alignment_check"] = True
            recovery_context["_recovery_prompt"] = (
                "Re-anchoring to goal. Perform alignment check before next action."
            )
            logger.info("goap_recovery_re_anchor", failures=self.failures)

        elif action == GOAPRecoveryAction.FALLBACK_AGENT:
            # Signal to use fallback agent
            recovery_context["_recovery_mode"] = "fallback_agent"
            recovery_context["_use_fallback"] = True
            logger.warning("goap_recovery_fallback_agent", failures=self.failures)

        elif action == GOAPRecoveryAction.REDUCE_SCOPE:
            # Reduce task scope
            recovery_context["_recovery_mode"] = "reduce_scope"
            recovery_context["_scope_reduced"] = True
            recovery_context["_recovery_prompt"] = (
                "Reducing task scope for recovery. Focus on minimum viable output."
            )
            logger.info("goap_recovery_reduce_scope", failures=self.failures)

        return recovery_context


# ========== LETTA MEMORY INTEGRATION ==========

class LettaMemoryBridge:
    """Bridge for sharing context across agents via Letta Cloud."""

    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or os.environ.get("UNLEASH_AGENT_ID")
        self._client = None

    @property
    def client(self):
        """Lazy-load Letta client."""
        if self._client is None:
            try:
                from letta_client import Letta
                api_key = os.environ.get("LETTA_API_KEY")
                if api_key:
                    self._client = Letta(
                        api_key=api_key,
                        base_url="https://api.letta.com"
                    )
            except ImportError:
                logger.warning("letta_client_not_installed")
        return self._client

    async def store_context(self, key: str, value: str, tags: Optional[list[str]] = None) -> bool:
        """Store context in Letta archival memory.

        V41 FIX: Letta SDK is synchronous - wrap with asyncio.to_thread() for async safety.
        """
        if not self.client or not self.agent_id:
            return False

        try:
            # V41: Wrap sync Letta call for async context
            await asyncio.to_thread(
                self.client.agents.passages.create,
                self.agent_id,
                text=f"[{key}] {value}",
                tags=tags or ["claude_flow_v3", "swarm_context"]
            )
            logger.info("letta_context_stored", key=key)
            return True
        except Exception as e:
            logger.warning("letta_store_failed", error=str(e))
            return False

    async def retrieve_context(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant context from Letta.

        V41 FIX: Letta SDK is synchronous - wrap with asyncio.to_thread() for async safety.
        """
        if not self.client or not self.agent_id:
            return []

        try:
            # V41: Wrap sync Letta call for async context
            results = await asyncio.to_thread(
                self.client.agents.passages.search,
                self.agent_id,
                query=query,
                top_k=top_k,
                tags=["claude_flow_v3"]
            )
            return [r.content for r in results.results]
        except Exception as e:
            logger.warning("letta_retrieve_failed", error=str(e))
            return []

    async def get_shared_block(self, block_label: str = "human") -> Optional[str]:
        """Get shared memory block value.

        V41 FIX: Letta SDK is synchronous - wrap with asyncio.to_thread() for async safety.
        """
        if not self.client or not self.agent_id:
            return None

        try:
            # V41: Wrap sync Letta call for async context
            block = await asyncio.to_thread(
                self.client.agents.blocks.retrieve,
                block_label,
                agent_id=self.agent_id
            )
            return block.value
        except Exception as e:
            logger.warning("letta_block_failed", error=str(e))
            return None


# ========== V3 ENHANCED AGENT ==========

@dataclass
class ClaudeAgentV3(ClaudeAgent):
    """
    V3 Enhanced Claude Agent with:
    - Circuit breaker for resilience
    - Confidence scoring
    - Self-critique capability
    """

    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    confidence_history: list[float] = field(default_factory=list)

    async def process_with_confidence(
        self,
        message: FlowMessage,
        context: dict[str, Any],
    ) -> tuple[AgentResult, float]:
        """Process with confidence score."""
        if not self.circuit_breaker.can_execute():
            logger.warning("circuit_breaker_blocked", agent=self.config.name)
            return AgentResult(
                agent_name=self.config.name,
                success=False,
                output="Circuit breaker open - agent temporarily unavailable",
                metadata={"circuit_state": "open"}
            ), 0.0

        try:
            result = await self.process(message, context)
            confidence = self._estimate_confidence(result.output)
            self.confidence_history.append(confidence)
            self.circuit_breaker.record_success()
            return result, confidence
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence from response content."""
        # Higher confidence for clear, structured responses
        confidence = 0.5

        # Increase for complete responses
        if "COMPLETE:" in response.upper():
            confidence += 0.3

        # Decrease for uncertainty indicators
        uncertainty_words = ["maybe", "possibly", "uncertain", "not sure", "might"]
        for word in uncertainty_words:
            if word in response.lower():
                confidence -= 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    async def self_critique(
        self,
        original_output: str,
        context: dict[str, Any],
    ) -> tuple[str, str]:
        """
        Self-critique the output and suggest improvements.
        Returns: (critique, improved_output)
        """
        critique_prompt = f"""## Self-Critique Task

Review your previous output and identify any issues or improvements.

### Previous Output
{original_output[:2000]}

### Context
{json.dumps(context, indent=2)}

### Instructions
1. Identify any errors, gaps, or areas for improvement
2. Rate the quality (1-10)
3. Provide a specific critique
4. Generate an improved version if needed

Format your response as:
QUALITY: [1-10]
CRITIQUE: [your critique]
IMPROVED: [improved output or "NO_CHANGE_NEEDED"]
"""

        critique_response = await self._call_llm(critique_prompt)

        # Parse critique
        critique = "Self-critique completed"
        improved = original_output

        if "CRITIQUE:" in critique_response:
            idx = critique_response.index("CRITIQUE:")
            end_idx = critique_response.find("IMPROVED:")
            if end_idx > idx:
                critique = critique_response[idx + 9:end_idx].strip()

        if "IMPROVED:" in critique_response and "NO_CHANGE_NEEDED" not in critique_response:
            idx = critique_response.index("IMPROVED:")
            improved = critique_response[idx + 9:].strip()

        return critique, improved


# ========== V3 FLOW ORCHESTRATOR ==========

@dataclass
class ClaudeFlowV3(ClaudeFlow):
    """
    Claude Flow V3 - Enhanced orchestration with swarm patterns.

    Key Enhancements:
    - Swarm spawning based on task complexity
    - Consensus-based decision making
    - Letta memory integration
    - Evaluator-Optimizer loops
    - Circuit breakers for resilience
    """

    swarm_config: SwarmConfig = field(default_factory=SwarmConfig)
    letta_bridge: Optional[LettaMemoryBridge] = None
    v3_agents: dict[str, ClaudeAgentV3] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize V3 components."""
        if self.swarm_config.enable_letta_memory:
            self.letta_bridge = LettaMemoryBridge(self.swarm_config.letta_agent_id)

    @classmethod
    def create_research_swarm(cls, name: str = "research-swarm") -> "ClaudeFlowV3":
        """Create a research-oriented swarm with multiple specialist agents."""
        agents = [
            AgentConfig(
                name="analyst",
                role=AgentRole.RESEARCHER,
                system_prompt="You are a research analyst. Find relevant information and data.",
                capabilities=["research", "analysis", "data_gathering"],
            ),
            AgentConfig(
                name="critic",
                role=AgentRole.CRITIC,
                system_prompt="You critically evaluate findings. Look for flaws and gaps.",
                capabilities=["critique", "validation", "fact_checking"],
            ),
            AgentConfig(
                name="synthesizer",
                role=AgentRole.COORDINATOR,
                system_prompt="You synthesize multiple perspectives into coherent insights.",
                capabilities=["synthesis", "summarization", "integration"],
            ),
        ]

        config = FlowConfig(
            name=name,
            agents=agents,
            entry_agent="analyst",
            parallel_execution=True,
        )

        swarm_config = SwarmConfig(
            strategy=SwarmStrategy.CONSENSUS,
            consensus_method=ConsensusMethod.MAJORITY,
            enable_letta_memory=True,
        )

        flow = cls(config=config, swarm_config=swarm_config)

        # Create V3 agents
        for agent_config in agents:
            flow.v3_agents[agent_config.name] = ClaudeAgentV3(config=agent_config)
            flow.agents[agent_config.name] = flow.v3_agents[agent_config.name]

        return flow

    @classmethod
    def create_evaluator_optimizer(cls, name: str = "eval-opt") -> "ClaudeFlowV3":
        """Create an evaluator-optimizer flow (Anthropic pattern)."""
        agents = [
            AgentConfig(
                name="generator",
                role=AgentRole.EXECUTOR,
                system_prompt="You generate solutions. Be creative and thorough.",
                capabilities=["generation", "creation"],
            ),
            AgentConfig(
                name="evaluator",
                role=AgentRole.REVIEWER,
                system_prompt="You evaluate solutions. Be critical and specific.",
                capabilities=["evaluation", "critique"],
            ),
        ]

        config = FlowConfig(
            name=name,
            agents=agents,
            entry_agent="generator",
        )

        swarm_config = SwarmConfig(
            strategy=SwarmStrategy.EVALUATOR_OPTIMIZER,
            evaluator_max_iterations=3,
        )

        flow = cls(config=config, swarm_config=swarm_config)

        for agent_config in agents:
            flow.v3_agents[agent_config.name] = ClaudeAgentV3(config=agent_config)
            flow.agents[agent_config.name] = flow.v3_agents[agent_config.name]

        return flow

    async def run_with_consensus(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        consensus_threshold: float = 0.5,
    ) -> FlowResult:
        """
        Run task with consensus-based decision making.
        Multiple agents vote on the output.
        """
        context = context or {}
        start_time = time.time()

        # Store task context in Letta
        if self.letta_bridge:
            await self.letta_bridge.store_context(
                key=f"task_{int(time.time())}",
                value=task,
                tags=["consensus_task"]
            )

        # Collect votes from all agents in parallel
        message = FlowMessage(
            type=MessageType.TASK,
            from_agent="orchestrator",
            content=task,
        )

        async def get_agent_vote(agent_name: str) -> ConsensusVote:
            agent = self.v3_agents.get(agent_name)
            if not agent:
                return ConsensusVote(
                    agent_name=agent_name,
                    decision="ABSTAIN",
                    confidence=0.0,
                    reasoning="Agent not found"
                )

            result, confidence = await agent.process_with_confidence(message, context)

            return ConsensusVote(
                agent_name=agent_name,
                decision=result.output[:500],  # Truncate for voting
                confidence=confidence,
                reasoning=f"Processed by {agent_name}"
            )

        # Run all agents in parallel
        votes = await asyncio.gather(*[
            get_agent_vote(name)
            for name in self.v3_agents.keys()
        ])

        # Tally votes
        consensus = self._tally_votes(list(votes), consensus_threshold)

        duration = time.time() - start_time

        return FlowResult(
            success=consensus.reached,
            final_output=consensus.winning_decision,
            total_steps=len(votes),
            duration_seconds=duration,
            metadata={"consensus": consensus.vote_count}
        )

    def _tally_votes(
        self,
        votes: list[ConsensusVote],
        threshold: float,
    ) -> ConsensusResult:
        """Tally votes and determine consensus."""
        if not votes:
            return ConsensusResult(reached=False, winning_decision=None)

        # Group by decision similarity (simplified - just use first N chars)
        vote_groups: dict[str, list[ConsensusVote]] = {}
        for vote in votes:
            key = vote.decision[:100]  # Group by first 100 chars
            if key not in vote_groups:
                vote_groups[key] = []
            vote_groups[key].append(vote)

        # Find winner
        total_votes = len(votes)
        vote_count = {k: len(v) for k, v in vote_groups.items()}
        winner_key = max(vote_groups.keys(), key=lambda k: len(vote_groups[k]))
        winner_votes = len(vote_groups[winner_key])

        # Check if consensus reached
        reached = (winner_votes / total_votes) >= threshold

        # Calculate total confidence for winner
        total_confidence = sum(v.confidence for v in vote_groups[winner_key])

        return ConsensusResult(
            reached=reached,
            winning_decision=vote_groups[winner_key][0].decision if reached else None,
            vote_count=vote_count,
            total_confidence=total_confidence / len(vote_groups[winner_key]),
            votes=votes,
            method=self.swarm_config.consensus_method,
        )

    async def run_evaluator_optimizer(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
    ) -> FlowResult:
        """
        Run evaluator-optimizer loop (Anthropic pattern).
        Generate → Evaluate → Feedback → Iterate until quality threshold.
        """
        context = context or {}
        start_time = time.time()

        generator = self.v3_agents.get("generator")
        evaluator = self.v3_agents.get("evaluator")

        if not generator or not evaluator:
            return FlowResult(
                success=False,
                error="Missing generator or evaluator agent",
                duration_seconds=time.time() - start_time,
            )

        current_output = ""
        iterations = []

        for i in range(max_iterations):
            logger.info("evaluator_optimizer_iteration", iteration=i + 1)

            # Generate
            gen_message = FlowMessage(
                type=MessageType.TASK,
                from_agent="orchestrator",
                content=f"{task}\n\nPrevious attempt: {current_output}" if current_output else task,
            )

            gen_result, gen_confidence = await generator.process_with_confidence(gen_message, context)
            current_output = gen_result.output

            # Evaluate
            eval_message = FlowMessage(
                type=MessageType.TASK,
                from_agent="generator",
                content=f"Evaluate this output:\n\n{current_output}\n\nOriginal task: {task}",
            )

            eval_result, eval_confidence = await evaluator.process_with_confidence(eval_message, context)

            iterations.append({
                "iteration": i + 1,
                "output": current_output[:200],
                "evaluation": eval_result.output[:200],
                "confidence": eval_confidence,
            })

            # Check if quality threshold met
            if eval_confidence >= quality_threshold:
                logger.info(
                    "evaluator_optimizer_converged",
                    iteration=i + 1,
                    confidence=eval_confidence,
                )
                break

            # Self-critique for next iteration
            critique, improved = await generator.self_critique(current_output, context)
            current_output = improved

        duration = time.time() - start_time

        return FlowResult(
            success=True,
            final_output=current_output,
            total_steps=len(iterations),
            duration_seconds=duration,
            metadata={"iterations": iterations}
        )

    async def run_parallel_swarm(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
    ) -> FlowResult:
        """
        Run all agents in parallel on the same task.
        Synthesize results at the end.
        """
        context = context or {}
        start_time = time.time()

        message = FlowMessage(
            type=MessageType.TASK,
            from_agent="orchestrator",
            content=task,
        )

        # Process all agents in parallel
        async def process_agent(name: str) -> tuple[str, AgentResult, float]:
            agent = self.v3_agents.get(name)
            if agent:
                result, confidence = await agent.process_with_confidence(message, context)
                return name, result, confidence
            return name, AgentResult(
                agent_name=name,
                success=False,
                output="Agent not found"
            ), 0.0

        results = await asyncio.gather(*[
            process_agent(name)
            for name in self.v3_agents.keys()
        ])

        # Synthesize results
        all_outputs = [
            f"[{name}] (confidence: {conf:.2f}): {result.output[:500]}"
            for name, result, conf in results
            if result.success
        ]

        synthesis = "\n\n".join(all_outputs)

        duration = time.time() - start_time

        return FlowResult(
            success=True,
            final_output=synthesis,
            total_steps=len(results),
            duration_seconds=duration,
            metadata={
                "agent_count": len(results),
                "confidences": {name: conf for name, _, conf in results}
            }
        )


# ========== FACTORY FUNCTIONS ==========

def create_v3_flow(
    name: str = "v3-flow",
    strategy: SwarmStrategy = SwarmStrategy.SEQUENTIAL,
) -> ClaudeFlowV3:
    """Factory function to create a V3 flow."""
    if strategy == SwarmStrategy.CONSENSUS:
        return ClaudeFlowV3.create_research_swarm(name)
    elif strategy == SwarmStrategy.EVALUATOR_OPTIMIZER:
        return ClaudeFlowV3.create_evaluator_optimizer(name)
    else:
        # Default to research swarm
        return ClaudeFlowV3.create_research_swarm(name)


# ========== CLI ==========

if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("Claude Flow V3 - Enhanced Multi-Agent Orchestration")
        print("=" * 60)

        # Test 1: Research Swarm with Consensus
        print("\n[Test 1] Research Swarm with Consensus...")
        swarm = ClaudeFlowV3.create_research_swarm()
        result = await swarm.run_with_consensus(
            "What are the key trends in AI for 2026?",
            consensus_threshold=0.5
        )
        print(f"  Consensus reached: {result.success}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Steps: {result.total_steps}")

        # Test 2: Evaluator-Optimizer Loop
        print("\n[Test 2] Evaluator-Optimizer Loop...")
        eval_opt = ClaudeFlowV3.create_evaluator_optimizer()
        result = await eval_opt.run_evaluator_optimizer(
            "Write a Python function to calculate Fibonacci numbers",
            max_iterations=2,
            quality_threshold=0.7
        )
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Iterations: {result.total_steps}")

        # Test 3: Parallel Swarm
        print("\n[Test 3] Parallel Swarm...")
        parallel = ClaudeFlowV3.create_research_swarm("parallel-test")
        result = await parallel.run_parallel_swarm(
            "Analyze the pros and cons of microservices architecture"
        )
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Agents: {result.metadata.get('agent_count', 0)}")

        print("\n" + "=" * 60)
        print("V3 Tests Complete")
        print("=" * 60)

    asyncio.run(main())
