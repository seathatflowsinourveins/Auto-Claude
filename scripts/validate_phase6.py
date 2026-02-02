#!/usr/bin/env python3
"""
Phase 6 Validation Script - Observability Layer
Tests all 6 SDK integrations with graceful fallback handling.
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ValidationResult:
    """Result of a validation test."""
    name: str
    passed: bool
    message: str
    sdk_available: bool = False


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def print_result(result: ValidationResult) -> None:
    """Print a validation result."""
    status = "[PASS]" if result.passed else "[FAIL]"
    sdk = "(SDK available)" if result.sdk_available else "(fallback mode)"
    print(f"  {status}: {result.name} {sdk}")
    if not result.passed:
        print(f"         {result.message}")


async def validate_langfuse() -> List[ValidationResult]:
    """Validate Langfuse tracer functionality."""
    results = []

    try:
        from core.observability import (
            LangfuseTracer,
            create_langfuse_tracer,
            LANGFUSE_AVAILABLE,
            TraceLevel,
            FeedbackType,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="Langfuse imports",
            passed=True,
            message="All exports accessible",
            sdk_available=LANGFUSE_AVAILABLE,
        ))

        # Test 2: Tracer creation
        tracer = create_langfuse_tracer()
        results.append(ValidationResult(
            name="LangfuseTracer creation",
            passed=tracer is not None,
            message="Created successfully" if tracer else "Failed to create",
            sdk_available=LANGFUSE_AVAILABLE,
        ))

        # Test 3: Start trace
        trace_result = tracer.start_trace("test-trace")
        results.append(ValidationResult(
            name="Start trace",
            passed=trace_result is not None,
            message="Trace started",
            sdk_available=LANGFUSE_AVAILABLE,
        ))

        # Test 4: Log generation
        gen_result = tracer.log_generation(
            trace_id=trace_result.trace_id,
            name="test-generation",
            prompt="Hello",
            completion="World",
            model="test-model",
        )
        results.append(ValidationResult(
            name="Log generation",
            passed=gen_result is not None,
            message="Generation logged",
            sdk_available=LANGFUSE_AVAILABLE,
        ))

        # Test 5: End trace
        final_result = tracer.end_trace(trace_result.trace_id)
        results.append(ValidationResult(
            name="End trace",
            passed=final_result is not None,
            message="Trace ended",
            sdk_available=LANGFUSE_AVAILABLE,
        ))

        # Test 6: Trace levels enum
        results.append(ValidationResult(
            name="TraceLevel enum",
            passed=len(TraceLevel) >= 4,
            message=f"Found {len(TraceLevel)} levels",
            sdk_available=LANGFUSE_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="Langfuse module",
            passed=False,
            message=str(e),
        ))

    return results


async def validate_opik() -> List[ValidationResult]:
    """Validate Opik evaluator functionality."""
    results = []

    try:
        from core.observability import (
            OpikEvaluator,
            create_opik_evaluator,
            OPIK_AVAILABLE,
            MetricType,
            EvaluationStatus,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="Opik imports",
            passed=True,
            message="All exports accessible",
            sdk_available=OPIK_AVAILABLE,
        ))

        # Test 2: Evaluator creation
        evaluator = create_opik_evaluator()
        results.append(ValidationResult(
            name="OpikEvaluator creation",
            passed=evaluator is not None,
            message="Created successfully",
            sdk_available=OPIK_AVAILABLE,
        ))

        # Test 3: Hallucination evaluation
        result = await evaluator.evaluate_hallucination(
            output="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital city.",
        )
        results.append(ValidationResult(
            name="Hallucination evaluation",
            passed=0.0 <= result.score <= 1.0,
            message=f"Score: {result.score:.3f}",
            sdk_available=OPIK_AVAILABLE,
        ))

        # Test 4: Relevance evaluation
        result = await evaluator.evaluate_relevance(
            input_text="What is the capital of France?",
            output="Paris is the capital of France.",
        )
        results.append(ValidationResult(
            name="Relevance evaluation",
            passed=0.0 <= result.score <= 1.0,
            message=f"Score: {result.score:.3f}",
            sdk_available=OPIK_AVAILABLE,
        ))

        # Test 5: MetricType enum
        results.append(ValidationResult(
            name="MetricType enum",
            passed=len(MetricType) >= 4,
            message=f"Found {len(MetricType)} types",
            sdk_available=OPIK_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="Opik module",
            passed=False,
            message=str(e),
        ))

    return results


async def validate_phoenix() -> List[ValidationResult]:
    """Validate Phoenix monitor functionality."""
    results = []

    try:
        from core.observability import (
            PhoenixMonitor,
            create_phoenix_monitor,
            PHOENIX_AVAILABLE,
            OPENTELEMETRY_AVAILABLE,
            AlertSeverity,
            DriftType,
            MonitorStatus,
            AlertConfig,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="Phoenix imports",
            passed=True,
            message="All exports accessible",
            sdk_available=PHOENIX_AVAILABLE,
        ))

        # Test 2: Monitor creation
        monitor = create_phoenix_monitor()
        results.append(ValidationResult(
            name="PhoenixMonitor creation",
            passed=monitor is not None,
            message="Created successfully",
            sdk_available=PHOENIX_AVAILABLE,
        ))

        # Test 3: Record request
        monitor.record_request(
            latency_ms=150.0,
            tokens=500,
            cost_usd=0.01,
            success=True,
        )
        results.append(ValidationResult(
            name="Record request",
            passed=True,
            message="Request recorded",
            sdk_available=PHOENIX_AVAILABLE,
        ))

        # Test 4: Get metrics
        metrics = monitor.get_metrics()
        results.append(ValidationResult(
            name="Get metrics",
            passed=metrics.total_requests >= 1,
            message=f"Requests: {metrics.total_requests}",
            sdk_available=PHOENIX_AVAILABLE,
        ))

        # Test 5: Add alert rule
        monitor.add_alert_rule(AlertConfig(
            name="high-latency",
            metric="latency_ms",
            threshold=1000.0,
            severity=AlertSeverity.WARNING,
        ))
        results.append(ValidationResult(
            name="Add alert rule",
            passed=True,
            message="Alert rule added",
            sdk_available=PHOENIX_AVAILABLE,
        ))

        # Test 6: Latency drift detection
        for _ in range(20):
            monitor.record_request(latency_ms=100.0, tokens=100, cost_usd=0.001)
        drift = monitor.detect_latency_drift()
        results.append(ValidationResult(
            name="Latency drift detection",
            passed=drift is not None,
            message=f"Drift score: {drift.score:.3f}",
            sdk_available=PHOENIX_AVAILABLE,
        ))

        # Test 7: OpenTelemetry availability
        results.append(ValidationResult(
            name="OpenTelemetry support",
            passed=True,
            message=f"Available: {OPENTELEMETRY_AVAILABLE}",
            sdk_available=OPENTELEMETRY_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="Phoenix module",
            passed=False,
            message=str(e),
        ))

    return results


async def validate_deepeval() -> List[ValidationResult]:
    """Validate DeepEval runner functionality."""
    results = []

    try:
        from core.observability import (
            DeepEvalRunner,
            create_deepeval_runner,
            DEEPEVAL_AVAILABLE,
            DeepEvalMetricType,
            TestStatus,
            DeepEvalTestCase,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="DeepEval imports",
            passed=True,
            message="All exports accessible",
            sdk_available=DEEPEVAL_AVAILABLE,
        ))

        # Test 2: Runner creation
        runner = create_deepeval_runner()
        results.append(ValidationResult(
            name="DeepEvalRunner creation",
            passed=runner is not None,
            message="Created successfully",
            sdk_available=DEEPEVAL_AVAILABLE,
        ))

        # Test 3: Faithfulness test
        result = await runner.test_faithfulness(
            input_text="What is Python?",
            output="Python is a programming language.",
            context="Python is a high-level programming language.",
        )
        results.append(ValidationResult(
            name="Faithfulness test",
            passed=0.0 <= result.score <= 1.0,
            message=f"Score: {result.score:.3f}",
            sdk_available=DEEPEVAL_AVAILABLE,
        ))

        # Test 4: Relevancy test
        result = await runner.test_relevancy(
            input_text="What is Python?",
            output="Python is a programming language.",
        )
        results.append(ValidationResult(
            name="Relevancy test",
            passed=0.0 <= result.score <= 1.0,
            message=f"Score: {result.score:.3f}",
            sdk_available=DEEPEVAL_AVAILABLE,
        ))

        # Test 5: Test suite execution
        test_cases = [
            DeepEvalTestCase(
                input_text="Hello",
                expected_output="Hi",
                actual_output="Hi there!",
            ),
        ]
        suite_result = await runner.run_test_suite(test_cases)
        results.append(ValidationResult(
            name="Test suite execution",
            passed=suite_result.total_tests >= 1,
            message=f"Tests run: {suite_result.total_tests}",
            sdk_available=DEEPEVAL_AVAILABLE,
        ))

        # Test 6: Metric types enum
        results.append(ValidationResult(
            name="DeepEvalMetricType enum",
            passed=len(DeepEvalMetricType) >= 5,
            message=f"Found {len(DeepEvalMetricType)} types",
            sdk_available=DEEPEVAL_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="DeepEval module",
            passed=False,
            message=str(e),
        ))

    return results


async def validate_ragas() -> List[ValidationResult]:
    """Validate RAGAS evaluator functionality."""
    results = []

    try:
        from core.observability import (
            RAGASEvaluator,
            create_ragas_evaluator,
            RAGAS_AVAILABLE,
            RAGASMetricType,
            RAGASSample,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="RAGAS imports",
            passed=True,
            message="All exports accessible",
            sdk_available=RAGAS_AVAILABLE,
        ))

        # Test 2: Evaluator creation
        evaluator = create_ragas_evaluator()
        results.append(ValidationResult(
            name="RAGASEvaluator creation",
            passed=evaluator is not None,
            message="Created successfully",
            sdk_available=RAGAS_AVAILABLE,
        ))

        # Test 3: Context precision
        result = await evaluator.evaluate_context_precision(
            question="What is the capital of France?",
            contexts=["France is a country. Paris is its capital."],
            ground_truth="Paris",
        )
        results.append(ValidationResult(
            name="Context precision",
            passed=0.0 <= result.score <= 1.0,
            message=f"Score: {result.score:.3f}",
            sdk_available=RAGAS_AVAILABLE,
        ))

        # Test 4: Faithfulness evaluation
        result = await evaluator.evaluate_faithfulness(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
        )
        results.append(ValidationResult(
            name="Faithfulness evaluation",
            passed=0.0 <= result.score <= 1.0,
            message=f"Score: {result.score:.3f}",
            sdk_available=RAGAS_AVAILABLE,
        ))

        # Test 5: Sample evaluation
        sample = RAGASSample(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["Artificial intelligence (AI) is computer-based intelligence."],
            ground_truth="Artificial intelligence",
        )
        sample_results = await evaluator.evaluate_sample(sample)
        results.append(ValidationResult(
            name="Sample evaluation",
            passed=len(sample_results) > 0,
            message=f"Metrics computed: {len(sample_results)}",
            sdk_available=RAGAS_AVAILABLE,
        ))

        # Test 6: Metric types enum
        results.append(ValidationResult(
            name="RAGASMetricType enum",
            passed=len(RAGASMetricType) >= 4,
            message=f"Found {len(RAGASMetricType)} types",
            sdk_available=RAGAS_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="RAGAS module",
            passed=False,
            message=str(e),
        ))

    return results


async def validate_promptfoo() -> List[ValidationResult]:
    """Validate Promptfoo runner functionality."""
    results = []

    try:
        from core.observability import (
            PromptfooRunner,
            create_promptfoo_runner,
            PROMPTFOO_AVAILABLE,
            RedTeamCategory,
            AssertionType,
            TestOutcome,
            PromptfooTestCase,
            PromptfooAssertion,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="Promptfoo imports",
            passed=True,
            message="All exports accessible",
            sdk_available=PROMPTFOO_AVAILABLE,
        ))

        # Test 2: Runner creation
        runner = create_promptfoo_runner()
        results.append(ValidationResult(
            name="PromptfooRunner creation",
            passed=runner is not None,
            message="Created successfully",
            sdk_available=PROMPTFOO_AVAILABLE,
        ))

        # Test 3: Red team test creation
        test = runner.create_red_team_test(RedTeamCategory.PROMPT_INJECTION)
        results.append(ValidationResult(
            name="Red team test creation",
            passed=test.attack_prompt != "",
            message=f"Category: {test.category.value}",
            sdk_available=PROMPTFOO_AVAILABLE,
        ))

        # Test 4: Run red team
        red_result = await runner.run_red_team(
            categories=[RedTeamCategory.PROMPT_INJECTION],
        )
        results.append(ValidationResult(
            name="Red team execution",
            passed=red_result.total_attacks > 0,
            message=f"Attacks: {red_result.total_attacks}, Block rate: {red_result.block_rate:.2%}",
            sdk_available=PROMPTFOO_AVAILABLE,
        ))

        # Test 5: Test case execution
        test_case = PromptfooTestCase(
            vars={"input": "Hello"},
            assertions=[
                PromptfooAssertion(type=AssertionType.CONTAINS, value=""),
            ],
        )
        test_result = await runner.run_test(test_case, "{{input}}")
        results.append(ValidationResult(
            name="Test case execution",
            passed=test_result is not None,
            message=f"Outcome: {test_result.outcome.value}",
            sdk_available=PROMPTFOO_AVAILABLE,
        ))

        # Test 6: Red team categories
        results.append(ValidationResult(
            name="RedTeamCategory enum",
            passed=len(RedTeamCategory) >= 4,
            message=f"Found {len(RedTeamCategory)} categories",
            sdk_available=PROMPTFOO_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="Promptfoo module",
            passed=False,
            message=str(e),
        ))

    return results


async def validate_factory() -> List[ValidationResult]:
    """Validate ObservabilityFactory functionality."""
    results = []

    try:
        from core.observability import (
            ObservabilityFactory,
            ObservabilityConfig,
            ObservabilityType,
            get_available_sdks,
            OBSERVABILITY_AVAILABLE,
        )

        # Test 1: Import check
        results.append(ValidationResult(
            name="Factory imports",
            passed=True,
            message="All exports accessible",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 2: Factory creation
        factory = ObservabilityFactory()
        results.append(ValidationResult(
            name="ObservabilityFactory creation",
            passed=factory is not None,
            message="Created successfully",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 3: Create tracer
        tracer = factory.create_tracer()
        results.append(ValidationResult(
            name="Factory.create_tracer()",
            passed=tracer is not None,
            message="Tracer created",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 4: Create evaluator
        evaluator = factory.create_evaluator()
        results.append(ValidationResult(
            name="Factory.create_evaluator()",
            passed=evaluator is not None,
            message="Evaluator created",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 5: Create monitor
        monitor = factory.create_monitor()
        results.append(ValidationResult(
            name="Factory.create_monitor()",
            passed=monitor is not None,
            message="Monitor created",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 6: Create tester
        tester = factory.create_tester()
        results.append(ValidationResult(
            name="Factory.create_tester()",
            passed=tester is not None,
            message="Tester created",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 7: Create RAG evaluator
        rag = factory.create_rag_evaluator()
        results.append(ValidationResult(
            name="Factory.create_rag_evaluator()",
            passed=rag is not None,
            message="RAG evaluator created",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 8: Create red teamer
        red = factory.create_red_teamer()
        results.append(ValidationResult(
            name="Factory.create_red_teamer()",
            passed=red is not None,
            message="Red teamer created",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 9: Get availability
        availability = factory.get_availability()
        results.append(ValidationResult(
            name="Factory.get_availability()",
            passed=len(availability) >= 6,
            message=f"SDKs tracked: {len(availability)}",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 10: Get available SDKs function
        sdks = get_available_sdks()
        results.append(ValidationResult(
            name="get_available_sdks()",
            passed=len(sdks) >= 6,
            message=f"SDKs: {sum(1 for v in sdks.values() if v)}/{len(sdks)} available",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

        # Test 11: ObservabilityType enum
        results.append(ValidationResult(
            name="ObservabilityType enum",
            passed=len(ObservabilityType) >= 6,
            message=f"Found {len(ObservabilityType)} types",
            sdk_available=OBSERVABILITY_AVAILABLE,
        ))

    except Exception as e:
        results.append(ValidationResult(
            name="Factory module",
            passed=False,
            message=str(e),
        ))

    return results


async def main() -> int:
    """Run all Phase 6 validations."""
    print_header("PHASE 6 VALIDATION - OBSERVABILITY LAYER")

    all_results: List[ValidationResult] = []

    # Run all validations
    validators = [
        ("Langfuse Tracer", validate_langfuse),
        ("Opik Evaluator", validate_opik),
        ("Phoenix Monitor", validate_phoenix),
        ("DeepEval Tests", validate_deepeval),
        ("RAGAS Evaluator", validate_ragas),
        ("Promptfoo Runner", validate_promptfoo),
        ("Observability Factory", validate_factory),
    ]

    for name, validator in validators:
        print(f"\n{name}:")
        print("-" * 40)
        results = await validator()
        all_results.extend(results)
        for result in results:
            print_result(result)

    # Summary
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    sdk_available = sum(1 for r in all_results if r.sdk_available)

    print(f"\n  Tests Passed: {passed}/{total}")
    print(f"  SDK Available: {sdk_available}/{total} tests have SDK backing")
    print(f"  Pass Rate: {passed/total*100:.1f}%")

    # SDK availability summary
    print("\n  SDK Availability:")
    try:
        from core.observability import get_available_sdks
        sdks = get_available_sdks()
        for sdk, available in sdks.items():
            status = "[Y]" if available else "[N]"
            print(f"    {status} {sdk}")
    except Exception:
        print("    (Could not load SDK availability)")

    if passed == total:
        print("\n" + "=" * 60)
        print(" [OK] PHASE 6 OBSERVABILITY LAYER: ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print(f" [FAIL] PHASE 6: {total - passed} TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
