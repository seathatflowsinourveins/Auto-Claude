"""
Braintrust Adapter - V36 Architecture

Integrates Braintrust for enterprise LLM evaluation and observability.

SDK: braintrust >= 0.1.0 (https://www.braintrust.dev/)
Layer: L5 (Observability)
Features:
- Real-time experiment tracking
- A/B testing for prompts
- Automatic scoring and metrics
- Dataset management
- Production monitoring
- Team collaboration

Braintrust Capabilities:
- Evals: Compare prompt/model performance
- Logging: Production trace capture
- Datasets: Versioned test sets
- Scores: Custom and built-in metrics

Usage:
    from platform.adapters.braintrust_adapter import BraintrustAdapter

    adapter = BraintrustAdapter()
    await adapter.initialize({"api_key": "...", "project": "my-project"})

    # Log an evaluation
    await adapter.execute("log", input="query", output="response", scores={"quality": 0.9})

    # Run experiment
    result = await adapter.execute("experiment", name="prompt-v2", data=[...])
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
BRAINTRUST_AVAILABLE = False

try:
    import braintrust
    BRAINTRUST_AVAILABLE = True
except ImportError:
    logger.info("Braintrust not installed - install with: pip install braintrust")


# Import base adapter interface
try:
    from platform.core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from enum import IntEnum
    from abc import ABC, abstractmethod

    class SDKLayer(IntEnum):
        OBSERVABILITY = 5

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0

    @_dataclass
    class AdapterConfig:
        name: str = "braintrust"
        layer: int = 5

    class AdapterStatus:
        READY = "ready"
        FAILED = "failed"
        UNINITIALIZED = "uninitialized"

    class SDKAdapter(ABC):
        @property
        @abstractmethod
        def sdk_name(self) -> str: ...
        @property
        @abstractmethod
        def layer(self) -> int: ...
        @property
        @abstractmethod
        def available(self) -> bool: ...
        @abstractmethod
        async def initialize(self, config: Dict) -> AdapterResult: ...
        @abstractmethod
        async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
        @abstractmethod
        async def health_check(self) -> AdapterResult: ...
        @abstractmethod
        async def shutdown(self) -> AdapterResult: ...


@dataclass
class EvalLog:
    """Evaluation log entry."""
    id: str
    input: Any
    output: Any
    expected: Optional[Any] = None
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExperimentResult:
    """Experiment result summary."""
    name: str
    total_samples: int
    avg_scores: Dict[str, float]
    comparison: Optional[Dict[str, Any]] = None


class BraintrustAdapter(SDKAdapter):
    """
    Braintrust adapter for LLM evaluation and observability.

    Provides enterprise-grade experiment tracking, A/B testing,
    and production monitoring for LLM applications.

    Operations:
    - log: Log a single evaluation
    - experiment: Run an experiment with multiple samples
    - dataset_create: Create a new dataset
    - dataset_insert: Insert data into dataset
    - compare: Compare two experiments
    - get_scores: Get score distributions
    - get_stats: Get adapter statistics
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="braintrust",
            layer=SDKLayer.OBSERVABILITY
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._project: str = ""
        self._api_key: str = ""
        self._logs: List[EvalLog] = []
        self._experiments: Dict[str, ExperimentResult] = {}
        self._datasets: Dict[str, List[Dict[str, Any]]] = {}
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    @property
    def sdk_name(self) -> str:
        return "braintrust"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.OBSERVABILITY

    @property
    def available(self) -> bool:
        return BRAINTRUST_AVAILABLE

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Braintrust adapter."""
        try:
            self._api_key = config.get("api_key") or os.environ.get("BRAINTRUST_API_KEY", "")
            self._project = config.get("project", "unleash-v36")

            if BRAINTRUST_AVAILABLE and self._api_key:
                # Initialize Braintrust client
                braintrust.login(api_key=self._api_key)

            self._status = AdapterStatus.READY
            logger.info(f"Braintrust adapter initialized (project={self._project})")

            return AdapterResult(
                success=True,
                data={
                    "project": self._project,
                    "has_api_key": bool(self._api_key),
                    "braintrust_native": BRAINTRUST_AVAILABLE
                }
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"Braintrust initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a Braintrust operation."""
        start_time = time.time()

        try:
            if operation == "log":
                result = await self._log(**kwargs)
            elif operation == "experiment":
                result = await self._experiment(**kwargs)
            elif operation == "dataset_create":
                result = await self._dataset_create(**kwargs)
            elif operation == "dataset_insert":
                result = await self._dataset_insert(**kwargs)
            elif operation == "compare":
                result = await self._compare(**kwargs)
            elif operation == "get_scores":
                result = await self._get_scores(**kwargs)
            elif operation == "get_stats":
                result = await self._get_stats()
            else:
                result = AdapterResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )

            latency_ms = (time.time() - start_time) * 1000
            self._call_count += 1
            self._total_latency_ms += latency_ms
            result.latency_ms = latency_ms

            if not result.success:
                self._error_count += 1

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Braintrust execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _log(
        self,
        input: Any,
        output: Any,
        expected: Optional[Any] = None,
        scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AdapterResult:
        """Log a single evaluation."""
        try:
            log_id = str(uuid.uuid4())

            log_entry = EvalLog(
                id=log_id,
                input=input,
                output=output,
                expected=expected,
                scores=scores or {},
                metadata=metadata or {}
            )
            self._logs.append(log_entry)

            if BRAINTRUST_AVAILABLE and self._api_key:
                # Log to Braintrust
                braintrust.log(
                    project=self._project,
                    input=input,
                    output=output,
                    expected=expected,
                    scores=scores,
                    metadata=metadata
                )

            return AdapterResult(
                success=True,
                data={
                    "log_id": log_id,
                    "scores": scores or {},
                    "logged": True
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _experiment(
        self,
        name: str,
        data: List[Dict[str, Any]],
        task: Optional[Callable] = None,
        scores: Optional[List[str]] = None,
        **kwargs
    ) -> AdapterResult:
        """Run an experiment with multiple samples."""
        try:
            all_scores: Dict[str, List[float]] = {}
            results = []

            for item in data:
                input_data = item.get("input")
                expected = item.get("expected")

                # Execute task if provided
                if task:
                    output = task(input_data)
                else:
                    output = item.get("output")

                # Calculate scores
                item_scores = item.get("scores", {})

                # Auto-calculate exact match if expected provided
                if expected is not None and "exact_match" not in item_scores:
                    item_scores["exact_match"] = 1.0 if output == expected else 0.0

                # Accumulate scores
                for score_name, score_value in item_scores.items():
                    if score_name not in all_scores:
                        all_scores[score_name] = []
                    all_scores[score_name].append(score_value)

                results.append({
                    "input": input_data,
                    "output": output,
                    "expected": expected,
                    "scores": item_scores
                })

            # Calculate averages
            avg_scores = {
                name: sum(values) / len(values)
                for name, values in all_scores.items()
            }

            experiment_result = ExperimentResult(
                name=name,
                total_samples=len(data),
                avg_scores=avg_scores
            )
            self._experiments[name] = experiment_result

            if BRAINTRUST_AVAILABLE and self._api_key:
                # Run Braintrust experiment
                with braintrust.Experiment(name=name, project=self._project) as exp:
                    for result in results:
                        exp.log(
                            input=result["input"],
                            output=result["output"],
                            expected=result["expected"],
                            scores=result["scores"]
                        )

            return AdapterResult(
                success=True,
                data={
                    "experiment_name": name,
                    "total_samples": len(data),
                    "avg_scores": avg_scores,
                    "results_preview": results[:3] if len(results) > 3 else results
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _dataset_create(
        self,
        name: str,
        description: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Create a new dataset."""
        try:
            self._datasets[name] = []

            if BRAINTRUST_AVAILABLE and self._api_key:
                braintrust.Dataset.create(
                    name=name,
                    project=self._project,
                    description=description
                )

            return AdapterResult(
                success=True,
                data={
                    "dataset_name": name,
                    "description": description,
                    "created": True
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _dataset_insert(
        self,
        name: str,
        records: List[Dict[str, Any]],
        **kwargs
    ) -> AdapterResult:
        """Insert records into a dataset."""
        try:
            if name not in self._datasets:
                self._datasets[name] = []

            self._datasets[name].extend(records)

            if BRAINTRUST_AVAILABLE and self._api_key:
                dataset = braintrust.Dataset.from_name(name, project=self._project)
                for record in records:
                    dataset.insert(**record)

            return AdapterResult(
                success=True,
                data={
                    "dataset_name": name,
                    "inserted": len(records),
                    "total_records": len(self._datasets[name])
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _compare(
        self,
        experiment_a: str,
        experiment_b: str,
        **kwargs
    ) -> AdapterResult:
        """Compare two experiments."""
        try:
            if experiment_a not in self._experiments:
                return AdapterResult(
                    success=False,
                    error=f"Experiment not found: {experiment_a}"
                )
            if experiment_b not in self._experiments:
                return AdapterResult(
                    success=False,
                    error=f"Experiment not found: {experiment_b}"
                )

            exp_a = self._experiments[experiment_a]
            exp_b = self._experiments[experiment_b]

            # Calculate differences
            comparison = {}
            all_scores = set(exp_a.avg_scores.keys()) | set(exp_b.avg_scores.keys())

            for score_name in all_scores:
                a_value = exp_a.avg_scores.get(score_name, 0.0)
                b_value = exp_b.avg_scores.get(score_name, 0.0)
                diff = b_value - a_value
                pct_change = (diff / a_value * 100) if a_value != 0 else 0

                comparison[score_name] = {
                    experiment_a: round(a_value, 4),
                    experiment_b: round(b_value, 4),
                    "difference": round(diff, 4),
                    "pct_change": round(pct_change, 2)
                }

            return AdapterResult(
                success=True,
                data={
                    "experiments": [experiment_a, experiment_b],
                    "comparison": comparison,
                    "winner": experiment_b if sum(
                        1 for c in comparison.values() if c["difference"] > 0
                    ) > len(comparison) / 2 else experiment_a
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_scores(
        self,
        experiment_name: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Get score distributions."""
        try:
            if experiment_name:
                if experiment_name not in self._experiments:
                    return AdapterResult(
                        success=False,
                        error=f"Experiment not found: {experiment_name}"
                    )
                experiments = {experiment_name: self._experiments[experiment_name]}
            else:
                experiments = self._experiments

            scores_data = {}
            for name, exp in experiments.items():
                scores_data[name] = {
                    "total_samples": exp.total_samples,
                    "avg_scores": exp.avg_scores
                }

            return AdapterResult(
                success=True,
                data={
                    "experiments": scores_data,
                    "count": len(experiments)
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_stats(self) -> AdapterResult:
        """Get adapter statistics."""
        return AdapterResult(
            success=True,
            data={
                "project": self._project,
                "total_logs": len(self._logs),
                "total_experiments": len(self._experiments),
                "total_datasets": len(self._datasets),
                "call_count": self._call_count,
                "error_count": self._error_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "braintrust_native": BRAINTRUST_AVAILABLE
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        return AdapterResult(
            success=True,
            data={
                "status": "healthy",
                "project": self._project,
                "has_api_key": bool(self._api_key),
                "braintrust_available": BRAINTRUST_AVAILABLE
            }
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._logs.clear()
        self._experiments.clear()
        self._datasets.clear()
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("Braintrust adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry
try:
    from platform.core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("braintrust", SDKLayer.OBSERVABILITY, priority=12)
    class RegisteredBraintrustAdapter(BraintrustAdapter):
        """Registered Braintrust adapter."""
        pass

except ImportError:
    pass


__all__ = ["BraintrustAdapter", "BRAINTRUST_AVAILABLE", "EvalLog", "ExperimentResult"]
