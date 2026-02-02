#!/usr/bin/env python3
"""
Opik V14 Enhanced Observability - Agent Optimizer & Guardrails

V14 enhancements over base opik_integration.py:
1. Agent Optimizer - Tracks multi-agent swarm patterns from Claude Flow V3
2. Guardrails - LLM output validation, safety checks, confidence thresholds
3. Enhanced Tracing - Hierarchical traces for consensus, parallel, and evaluator-optimizer patterns
4. Cost Optimization - Budget tracking with automatic model routing

Integrates with:
- claude_flow_v3.py (SwarmStrategy, ConsensusMethod, ClaudeAgentV3)
- opik_evaluator.py (MetricType, EvaluationResult)
- Letta agents (sleep-time, memory persistence)

Usage:
    from core.opik_v14 import OpikV14, AgentOptimizer, OutputGuardrail

    optimizer = AgentOptimizer()
    guardrail = OutputGuardrail()

    # Trace multi-agent execution
    async with optimizer.trace_swarm("consensus", agents=["a1", "a2", "a3"]) as ctx:
        results = await run_consensus(agents)
        ctx.record_results(results)

    # Validate LLM outputs
    validation = await guardrail.validate(output, context=context)
    if not validation.passed:
        output = await guardrail.remediate(output, validation.issues)
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# Import base Opik integration - handle different import paths
# Python's built-in 'platform' module shadows our directory, so we use multiple fallbacks
try:
    from core.opik_integration import (
        OpikClient,
        OpikConfig,
        get_opik_client,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from .opik_integration import (
            OpikClient,
            OpikConfig,
            get_opik_client,
        )
    except ImportError:
        # Final fallback: load directly from file path
        import importlib.util
        import os
        import sys as _sys
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _opik_integration_path = os.path.join(_current_dir, "opik_integration.py")
        if os.path.exists(_opik_integration_path):
            _module_name = "opik_integration_direct"
            _spec = importlib.util.spec_from_file_location(_module_name, _opik_integration_path)
            if _spec and _spec.loader:
                _opik_module = importlib.util.module_from_spec(_spec)
                # CRITICAL: Register in sys.modules BEFORE exec_module for dataclass compatibility
                _sys.modules[_module_name] = _opik_module
                _spec.loader.exec_module(_opik_module)
                OpikClient = _opik_module.OpikClient
                OpikConfig = _opik_module.OpikConfig
                get_opik_client = _opik_module.get_opik_client
            else:
                raise ImportError("Cannot load opik_integration module")
        else:
            raise ImportError(f"opik_integration.py not found at {_opik_integration_path}")

try:
    from core.observability.opik_evaluator import (
        OpikEvaluator,
    )
except ImportError:
    # Fallback: create minimal evaluator if not available
    class OpikEvaluator:  # type: ignore
        """Minimal fallback evaluator."""
        def __init__(self, **_kwargs):
            pass

        async def evaluate_hallucination(self, output: str, context: str, threshold: float = 0.5):
            # Simple heuristic fallback
            output_words = set(output.lower().split())
            context_words = set(context.lower().split())
            overlap = len(output_words & context_words)
            score = 1.0 - min(1.0, (overlap / len(output_words) * 1.5)) if output_words else 0.5
            return type("Result", (), {"passed": score <= threshold, "score": score})()

        async def evaluate_relevance(self, input_text: str, output: str, threshold: float = 0.7):
            input_words = set(input_text.lower().split())
            output_words = set(output.lower().split())
            overlap = len(input_words & output_words)
            score = min(1.0, overlap / len(input_words) * 2) if input_words else 0.5
            return type("Result", (), {"passed": score >= threshold, "score": score})()

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# V14 Enums and Models
# =============================================================================

class SwarmTraceType(str, Enum):
    """Types of swarm execution patterns to trace."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    HIERARCHICAL = "hierarchical"


class GuardrailType(str, Enum):
    """Types of output guardrails."""
    HALLUCINATION = "hallucination"
    RELEVANCE = "relevance"
    TOXICITY = "toxicity"
    PII = "pii"
    CONFIDENCE = "confidence"
    FORMAT = "format"
    LENGTH = "length"
    CUSTOM = "custom"


class RemediationStrategy(str, Enum):
    """Strategies for remediating guardrail violations."""
    REJECT = "reject"
    RETRY = "retry"
    FILTER = "filter"
    TRANSFORM = "transform"
    FALLBACK = "fallback"


@dataclass
class GuardrailViolation:
    """A single guardrail violation."""
    guardrail_type: GuardrailType
    severity: str = "warning"  # "info", "warning", "error", "critical"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[RemediationStrategy] = None


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmTrace(BaseModel):
    """Trace for multi-agent swarm execution."""
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    swarm_type: SwarmTraceType
    agent_ids: List[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    total_latency_ms: float = 0.0

    # Agent-level metrics
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agent_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    agent_costs_usd: Dict[str, float] = Field(default_factory=dict)

    # Swarm-level metrics
    consensus_rounds: int = 0
    optimizer_iterations: int = 0
    final_confidence: float = 0.0
    winning_agent: Optional[str] = None

    # Cost tracking
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for a single agent over time."""
    agent_id: str
    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    avg_confidence: float = 0.0
    consensus_win_rate: float = 0.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Agent Optimizer
# =============================================================================

class AgentOptimizer:
    """
    Tracks and optimizes multi-agent swarm execution patterns.

    Features:
    - Hierarchical tracing for swarm patterns
    - Per-agent performance tracking
    - Cost optimization with budget alerts
    - Automatic model routing based on performance
    """

    def __init__(
        self,
        opik_client: Optional[OpikClient] = None,
        budget_limit_usd: float = 10.0,
        enable_auto_routing: bool = True,
    ):
        self.opik = opik_client or get_opik_client()
        self.budget_limit_usd = budget_limit_usd
        self.enable_auto_routing = enable_auto_routing

        # Performance tracking
        self._swarm_traces: List[SwarmTrace] = []
        self._agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self._session_cost_usd: float = 0.0

        # Model routing preferences
        self._model_preferences: Dict[str, str] = {
            "high_confidence": "claude-opus-4-5",
            "balanced": "claude-3-5-sonnet",
            "cost_efficient": "claude-3-haiku",
        }

    @asynccontextmanager
    async def trace_swarm(
        self,
        swarm_type: Union[SwarmTraceType, str],
        agent_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Async context manager for tracing swarm execution.

        Usage:
            async with optimizer.trace_swarm("consensus", ["a1", "a2"]) as ctx:
                results = await execute_consensus()
                ctx.record_results(results)
        """
        if isinstance(swarm_type, str):
            swarm_type = SwarmTraceType(swarm_type)

        trace = SwarmTrace(
            swarm_type=swarm_type,
            agent_ids=agent_ids,
            metadata=metadata or {},
        )

        start_time = time.perf_counter()

        class TraceContext:
            def __init__(self, trace_ref: SwarmTrace, optimizer_ref: AgentOptimizer):
                self.trace = trace_ref
                self.optimizer = optimizer_ref

            def record_agent_result(
                self,
                agent_id: str,
                result: Any,
                latency_ms: float = 0.0,
                cost_usd: float = 0.0,
                tokens: int = 0,
            ):
                """Record result from a single agent."""
                self.trace.agent_results[agent_id] = {
                    "result": str(result)[:500] if result else None,
                    "success": True,
                }
                self.trace.agent_latencies_ms[agent_id] = latency_ms
                self.trace.agent_costs_usd[agent_id] = cost_usd
                self.trace.total_tokens += tokens
                self.trace.total_cost_usd += cost_usd

            def record_consensus_round(self, round_num: int, agreement_score: float):
                """Record a consensus round."""
                self.trace.consensus_rounds = round_num
                self.trace.final_confidence = agreement_score

            def record_optimizer_iteration(
                self,
                iteration: int,
                score: float,
                feedback: str,
            ):
                """Record an evaluator-optimizer iteration."""
                self.trace.optimizer_iterations = iteration
                if "iterations" not in self.trace.metadata:
                    self.trace.metadata["iterations"] = []
                self.trace.metadata["iterations"].append({
                    "iteration": iteration,
                    "score": score,
                    "feedback": feedback[:200],
                })

            def set_winner(self, agent_id: str, confidence: float = 1.0):
                """Set the winning agent for consensus."""
                self.trace.winning_agent = agent_id
                self.trace.final_confidence = confidence

            def record_error(self, error: str):
                """Record an error during swarm execution."""
                self.trace.error = error

        ctx = TraceContext(trace, self)

        try:
            yield ctx
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            # Finalize trace
            trace.end_time = datetime.now(timezone.utc)
            trace.total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Update session cost
            self._session_cost_usd += trace.total_cost_usd

            # Store trace
            self._swarm_traces.append(trace)

            # Update agent metrics
            for agent_id in agent_ids:
                self._update_agent_metrics(agent_id, trace)

            # Log to Opik if connected
            self._log_swarm_trace(trace)

            # Budget alert
            if self._session_cost_usd >= self.budget_limit_usd * 0.8:
                logger.warning(
                    f"Budget alert: {self._session_cost_usd:.4f} / {self.budget_limit_usd:.4f} USD"
                )

    @contextmanager
    def trace_swarm_sync(
        self,
        swarm_type: Union[SwarmTraceType, str],
        agent_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Synchronous version of trace_swarm."""
        if isinstance(swarm_type, str):
            swarm_type = SwarmTraceType(swarm_type)

        trace = SwarmTrace(
            swarm_type=swarm_type,
            agent_ids=agent_ids,
            metadata=metadata or {},
        )

        start_time = time.perf_counter()

        class TraceContextSync:
            def __init__(self, trace_ref: SwarmTrace):
                self.trace = trace_ref

            def record_agent_result(
                self,
                agent_id: str,
                result: Any,
                latency_ms: float = 0.0,
                cost_usd: float = 0.0,
                tokens: int = 0,
            ):
                self.trace.agent_results[agent_id] = {
                    "result": str(result)[:500] if result else None,
                    "success": True,
                }
                self.trace.agent_latencies_ms[agent_id] = latency_ms
                self.trace.agent_costs_usd[agent_id] = cost_usd
                self.trace.total_tokens += tokens
                self.trace.total_cost_usd += cost_usd

            def set_winner(self, agent_id: str, confidence: float = 1.0):
                self.trace.winning_agent = agent_id
                self.trace.final_confidence = confidence

        ctx = TraceContextSync(trace)

        try:
            yield ctx
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            trace.end_time = datetime.now(timezone.utc)
            trace.total_latency_ms = (time.perf_counter() - start_time) * 1000
            self._session_cost_usd += trace.total_cost_usd
            self._swarm_traces.append(trace)
            self._log_swarm_trace(trace)

    def _update_agent_metrics(self, agent_id: str, trace: SwarmTrace) -> None:
        """Update performance metrics for an agent."""
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentPerformanceMetrics(agent_id=agent_id)

        metrics = self._agent_metrics[agent_id]
        metrics.total_calls += 1
        metrics.total_cost_usd += trace.agent_costs_usd.get(agent_id, 0.0)
        metrics.total_tokens += trace.total_tokens // len(trace.agent_ids) if trace.agent_ids else 0

        # Update latency (rolling average)
        latency = trace.agent_latencies_ms.get(agent_id, 0.0)
        metrics.avg_latency_ms = (
            (metrics.avg_latency_ms * (metrics.total_calls - 1) + latency)
            / metrics.total_calls
        )

        # Update consensus win rate
        if trace.winning_agent == agent_id:
            wins = metrics.consensus_win_rate * (metrics.total_calls - 1) + 1
            metrics.consensus_win_rate = wins / metrics.total_calls

        metrics.last_updated = datetime.now(timezone.utc)

    def _log_swarm_trace(self, trace: SwarmTrace) -> None:
        """Log swarm trace to Opik."""
        if not self.opik.is_connected:
            return

        self.opik.create_trace(
            model=f"swarm:{trace.swarm_type.value}",
            provider="claude-flow-v3",
            operation=f"swarm.{trace.swarm_type.value}",
            input_text=json.dumps({"agent_ids": trace.agent_ids}),
            output_text=json.dumps({
                "winner": trace.winning_agent,
                "confidence": trace.final_confidence,
            }),
            latency_ms=trace.total_latency_ms,
            error=trace.error,
            metadata={
                "swarm_type": trace.swarm_type.value,
                "agent_count": len(trace.agent_ids),
                "consensus_rounds": trace.consensus_rounds,
                "optimizer_iterations": trace.optimizer_iterations,
                **trace.metadata,
            },
        )

    def get_recommended_model(self, task_type: str = "balanced") -> str:
        """Get recommended model based on budget and performance."""
        if self._session_cost_usd >= self.budget_limit_usd * 0.9:
            return self._model_preferences["cost_efficient"]
        elif task_type in self._model_preferences:
            return self._model_preferences[task_type]
        return self._model_preferences["balanced"]

    def get_agent_stats(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """Get performance stats for an agent."""
        return self._agent_metrics.get(agent_id)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session-level statistics."""
        if not self._swarm_traces:
            return {
                "total_traces": 0,
                "total_cost_usd": 0.0,
                "budget_remaining_usd": self.budget_limit_usd,
            }

        return {
            "total_traces": len(self._swarm_traces),
            "total_cost_usd": round(self._session_cost_usd, 4),
            "budget_remaining_usd": round(self.budget_limit_usd - self._session_cost_usd, 4),
            "budget_usage_pct": round(self._session_cost_usd / self.budget_limit_usd * 100, 1),
            "by_swarm_type": self._stats_by_swarm_type(),
            "agent_count": len(self._agent_metrics),
            "avg_latency_ms": self._avg_trace_latency(),
        }

    def _stats_by_swarm_type(self) -> Dict[str, int]:
        """Get trace count by swarm type."""
        by_type: Dict[str, int] = {}
        for trace in self._swarm_traces:
            key = trace.swarm_type.value
            by_type[key] = by_type.get(key, 0) + 1
        return by_type

    def _avg_trace_latency(self) -> float:
        """Get average trace latency."""
        if not self._swarm_traces:
            return 0.0
        total = sum(t.total_latency_ms for t in self._swarm_traces)
        return round(total / len(self._swarm_traces), 2)


# =============================================================================
# Output Guardrails
# =============================================================================

class OutputGuardrail:
    """
    LLM output validation and safety guardrails.

    Features:
    - Hallucination detection
    - Relevance validation
    - PII detection
    - Confidence thresholds
    - Format validation
    - Automatic remediation
    """

    def __init__(
        self,
        evaluator: Optional[OpikEvaluator] = None,
        hallucination_threshold: float = 0.5,
        relevance_threshold: float = 0.7,
        confidence_threshold: float = 0.6,
        enable_pii_detection: bool = True,
        enable_toxicity_check: bool = True,
    ):
        self.evaluator = evaluator or OpikEvaluator()
        self.hallucination_threshold = hallucination_threshold
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_pii_detection = enable_pii_detection
        self.enable_toxicity_check = enable_toxicity_check

        # PII patterns (basic)
        self._pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        ]

        # Custom guardrails registry
        self._custom_guardrails: Dict[str, Callable[[str, Dict], GuardrailViolation | None]] = {}

    async def validate(
        self,
        output: str,
        input_text: Optional[str] = None,
        context: Optional[str] = None,
        expected_format: Optional[str] = None,
        max_length: Optional[int] = None,
        custom_checks: Optional[List[str]] = None,
    ) -> GuardrailResult:
        """
        Validate LLM output against all configured guardrails.

        Args:
            output: The LLM output to validate
            input_text: Original input (for relevance check)
            context: Context (for hallucination check)
            expected_format: Expected format (json, markdown, etc.)
            max_length: Maximum allowed length
            custom_checks: List of custom guardrail names to run

        Returns:
            GuardrailResult with pass/fail and any violations
        """
        start_time = time.perf_counter()
        violations: List[GuardrailViolation] = []

        # Run checks in parallel where possible
        tasks = []

        # Hallucination check
        if context:
            tasks.append(self._check_hallucination(output, context))

        # Relevance check
        if input_text:
            tasks.append(self._check_relevance(output, input_text))

        # PII check
        if self.enable_pii_detection:
            tasks.append(self._check_pii(output))

        # Format check
        if expected_format:
            tasks.append(self._check_format(output, expected_format))

        # Length check
        if max_length:
            tasks.append(self._check_length(output, max_length))

        # Custom checks
        if custom_checks:
            for check_name in custom_checks:
                if check_name in self._custom_guardrails:
                    tasks.append(self._run_custom_check(check_name, output, {
                        "input": input_text,
                        "context": context,
                    }))

        # Execute all checks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, GuardrailViolation):
                    violations.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Guardrail check failed: {result}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Determine overall pass/fail
        has_critical = any(v.severity == "critical" for v in violations)
        has_error = any(v.severity == "error" for v in violations)

        return GuardrailResult(
            passed=not (has_critical or has_error),
            violations=violations,
            confidence=1.0 - (len(violations) * 0.1),  # Simple confidence decay
            latency_ms=latency_ms,
            metadata={
                "checks_run": len(tasks),
                "violations_count": len(violations),
            },
        )

    async def _check_hallucination(
        self,
        output: str,
        context: str,
    ) -> Optional[GuardrailViolation]:
        """Check for hallucination."""
        result = await self.evaluator.evaluate_hallucination(
            output, context, threshold=self.hallucination_threshold
        )

        if not result.passed:
            return GuardrailViolation(
                guardrail_type=GuardrailType.HALLUCINATION,
                severity="error" if result.score > 0.7 else "warning",
                message=f"Hallucination detected (score: {result.score:.2f})",
                details={"score": result.score, "threshold": self.hallucination_threshold},
                remediation=RemediationStrategy.RETRY,
            )
        return None

    async def _check_relevance(
        self,
        output: str,
        input_text: str,
    ) -> Optional[GuardrailViolation]:
        """Check output relevance to input."""
        result = await self.evaluator.evaluate_relevance(
            input_text, output, threshold=self.relevance_threshold
        )

        if not result.passed:
            return GuardrailViolation(
                guardrail_type=GuardrailType.RELEVANCE,
                severity="warning",
                message=f"Low relevance (score: {result.score:.2f})",
                details={"score": result.score, "threshold": self.relevance_threshold},
                remediation=RemediationStrategy.RETRY,
            )
        return None

    async def _check_pii(self, output: str) -> Optional[GuardrailViolation]:
        """Check for PII in output."""
        found_pii = []
        for pattern in self._pii_patterns:
            if re.search(pattern, output):
                found_pii.append(pattern)

        if found_pii:
            return GuardrailViolation(
                guardrail_type=GuardrailType.PII,
                severity="critical",
                message=f"PII detected in output ({len(found_pii)} patterns matched)",
                details={"patterns_matched": len(found_pii)},
                remediation=RemediationStrategy.FILTER,
            )
        return None

    async def _check_format(
        self,
        output: str,
        expected_format: str,
    ) -> Optional[GuardrailViolation]:
        """Check output format."""
        if expected_format == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError as e:
                return GuardrailViolation(
                    guardrail_type=GuardrailType.FORMAT,
                    severity="error",
                    message=f"Invalid JSON format: {str(e)[:100]}",
                    details={"expected": "json", "error": str(e)[:200]},
                    remediation=RemediationStrategy.TRANSFORM,
                )
        elif expected_format == "markdown":
            # Basic markdown validation
            if not any(marker in output for marker in ["#", "-", "*", "```", "**"]):
                return GuardrailViolation(
                    guardrail_type=GuardrailType.FORMAT,
                    severity="info",
                    message="Output doesn't appear to be markdown formatted",
                    details={"expected": "markdown"},
                    remediation=RemediationStrategy.TRANSFORM,
                )
        return None

    async def _check_length(
        self,
        output: str,
        max_length: int,
    ) -> Optional[GuardrailViolation]:
        """Check output length."""
        if len(output) > max_length:
            return GuardrailViolation(
                guardrail_type=GuardrailType.LENGTH,
                severity="warning",
                message=f"Output exceeds max length ({len(output)} > {max_length})",
                details={"actual_length": len(output), "max_length": max_length},
                remediation=RemediationStrategy.FILTER,
            )
        return None

    async def _run_custom_check(
        self,
        check_name: str,
        output: str,
        context: Dict[str, Any],
    ) -> Optional[GuardrailViolation]:
        """Run a custom guardrail check."""
        if check_name not in self._custom_guardrails:
            return None

        check_fn = self._custom_guardrails[check_name]
        try:
            return check_fn(output, context)
        except Exception as e:
            logger.warning(f"Custom guardrail {check_name} failed: {e}")
            return None

    def register_custom_guardrail(
        self,
        name: str,
        check_fn: Callable[[str, Dict], GuardrailViolation | None],
    ) -> None:
        """Register a custom guardrail check."""
        self._custom_guardrails[name] = check_fn

    async def remediate(
        self,
        output: str,
        violations: List[GuardrailViolation],
        _max_retries: int = 2,  # Reserved for future retry logic
    ) -> str:
        """
        Attempt to remediate guardrail violations.

        Args:
            output: Original output
            violations: List of violations to remediate
            max_retries: Maximum remediation attempts

        Returns:
            Remediated output (or original if remediation fails)
        """
        remediated = output

        for violation in violations:
            if violation.remediation == RemediationStrategy.FILTER:
                # Filter out problematic content
                remediated = self._filter_content(remediated, violation)
            elif violation.remediation == RemediationStrategy.TRANSFORM:
                # Transform to expected format
                remediated = self._transform_format(remediated, violation)
            # RETRY and REJECT should be handled by caller

        return remediated

    def _filter_content(self, output: str, violation: GuardrailViolation) -> str:
        """Filter problematic content from output."""
        import re

        if violation.guardrail_type == GuardrailType.PII:
            # Redact PII patterns
            for pattern in self._pii_patterns:
                output = re.sub(pattern, "[REDACTED]", output)
        elif violation.guardrail_type == GuardrailType.LENGTH:
            # Truncate to max length
            max_len = violation.details.get("max_length", 1000)
            if len(output) > max_len:
                output = output[:max_len] + "..."

        return output

    def _transform_format(self, output: str, violation: GuardrailViolation) -> str:
        """Transform output to expected format."""
        expected = violation.details.get("expected", "")

        if expected == "json":
            # Attempt to extract JSON from output
            try:
                # Look for JSON block
                import re
                json_match = re.search(r'```json?\s*([\s\S]*?)```', output)
                if json_match:
                    json.loads(json_match.group(1))
                    return json_match.group(1)

                # Try to wrap as string
                return json.dumps({"content": output})
            except Exception:
                return output

        return output


# =============================================================================
# Integrated V14 Client
# =============================================================================

class OpikV14:
    """
    Integrated V14 Opik client with agent optimizer and guardrails.

    Combines:
    - Base OpikClient for LLM tracing
    - AgentOptimizer for swarm patterns
    - OutputGuardrail for safety validation
    """

    def __init__(
        self,
        config: Optional[OpikConfig] = None,
        budget_limit_usd: float = 10.0,
        enable_guardrails: bool = True,
    ):
        self.opik = get_opik_client(config)
        self.optimizer = AgentOptimizer(
            opik_client=self.opik,
            budget_limit_usd=budget_limit_usd,
        )
        self.guardrail = OutputGuardrail() if enable_guardrails else None
        self.evaluator = OpikEvaluator()

    @asynccontextmanager
    async def trace_agent_call(
        self,
        agent_id: str,
        model: str,
        operation: str,
        validate_output: bool = True,
        context: Optional[str] = None,
    ):
        """
        Trace a single agent call with optional guardrail validation.

        Usage:
            async with v14.trace_agent_call("agent-1", "claude-3-5-sonnet", "generate") as ctx:
                response = await agent.generate(prompt)
                ctx.set_output(response)
                # Output is automatically validated
        """
        start_time = time.perf_counter()
        result: Dict[str, Any] = {
            "input": None,
            "output": None,
            "tokens": 0,
            "validated": False,
            "guardrail_result": None,
        }

        class CallContext:
            def __init__(self, result_ref: Dict, guardrail_ref: Optional[OutputGuardrail]):
                self._result = result_ref
                self._guardrail = guardrail_ref
                self._context = context

            def set_input(self, input_text: str):
                self._result["input"] = input_text

            async def set_output(self, output: str, validate: bool = True):
                self._result["output"] = output
                if validate and self._guardrail:
                    gr_result = await self._guardrail.validate(
                        output,
                        input_text=self._result["input"],
                        context=self._context,
                    )
                    self._result["validated"] = True
                    self._result["guardrail_result"] = gr_result

            def set_tokens(self, input_tokens: int, output_tokens: int):
                self._result["tokens"] = input_tokens + output_tokens

        ctx = CallContext(result, self.guardrail if validate_output else None)

        try:
            yield ctx
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log to base Opik
            error = None
            if result["guardrail_result"] and not result["guardrail_result"].passed:
                violations = result["guardrail_result"].violations
                error = f"Guardrail violations: {len(violations)}"

            self.opik.create_trace(
                model=model,
                provider="v14-agent",
                operation=operation,
                input_text=result["input"],
                output_text=result["output"],
                latency_ms=latency_ms,
                error=error,
                metadata={
                    "agent_id": agent_id,
                    "validated": result["validated"],
                    "guardrail_passed": result["guardrail_result"].passed if result["guardrail_result"] else None,
                },
            )

    def trace_swarm(self, *args, **kwargs):
        """Delegate to optimizer's trace_swarm."""
        return self.optimizer.trace_swarm(*args, **kwargs)

    async def validate_output(self, *args, **kwargs) -> GuardrailResult:
        """Delegate to guardrail's validate."""
        if not self.guardrail:
            return GuardrailResult(passed=True)
        return await self.guardrail.validate(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "opik": self.opik.get_session_stats(),
            "optimizer": self.optimizer.get_session_stats(),
        }


# =============================================================================
# Decorator for V14 tracing
# =============================================================================

def trace_v14(
    model: str = "claude-3-5-sonnet",
    operation: Optional[str] = None,
    validate: bool = True,
):
    """
    Decorator for V14 tracing with guardrails.

    Usage:
        @trace_v14(model="claude-3-5-sonnet", validate=True)
        async def my_agent_function(prompt: str) -> str:
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            v14 = _get_v14_instance()
            agent_id = kwargs.get("agent_id", "default")

            async with v14.trace_agent_call(agent_id, model, op_name, validate_output=validate) as ctx:
                # Capture input
                if args and isinstance(args[0], str):
                    ctx.set_input(args[0])
                elif "prompt" in kwargs:
                    ctx.set_input(str(kwargs["prompt"]))

                result = await func(*args, **kwargs)

                # Capture and validate output
                if isinstance(result, str):
                    await ctx.set_output(result)

                return result

        return async_wrapper  # type: ignore

    return decorator


# =============================================================================
# Global Instance
# =============================================================================

_v14_instance: Optional[OpikV14] = None


def _get_v14_instance() -> OpikV14:
    """Get or create global V14 instance."""
    global _v14_instance
    if _v14_instance is None:
        _v14_instance = OpikV14()
    return _v14_instance


def configure_v14(
    budget_limit_usd: float = 10.0,
    enable_guardrails: bool = True,
) -> OpikV14:
    """Configure and return the V14 Opik client."""
    global _v14_instance
    _v14_instance = OpikV14(
        budget_limit_usd=budget_limit_usd,
        enable_guardrails=enable_guardrails,
    )
    return _v14_instance


def get_v14_client() -> OpikV14:
    """Get the global V14 client instance."""
    return _get_v14_instance()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # V14 Client
    "OpikV14",
    "configure_v14",
    "get_v14_client",

    # Agent Optimizer
    "AgentOptimizer",
    "SwarmTraceType",
    "SwarmTrace",
    "AgentPerformanceMetrics",

    # Guardrails
    "OutputGuardrail",
    "GuardrailType",
    "GuardrailViolation",
    "GuardrailResult",
    "RemediationStrategy",

    # Decorator
    "trace_v14",
]
