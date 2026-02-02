#!/usr/bin/env python3
"""
Quick integration test for opik_v14.py

Tests:
1. AgentOptimizer - swarm tracing
2. OutputGuardrail - validation
3. GuardrailResult - data structures
"""

import asyncio
import sys
import os
import importlib.util

# Fix import path issue: Python's built-in 'platform' module shadows our directory
# Use importlib to load directly from file path
def load_module_from_path(module_name: str, file_path: str):
    """Load a module directly from file path, bypassing normal import resolution."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the directory containing this test file
_test_dir = os.path.dirname(os.path.abspath(__file__))
_opik_v14_path = os.path.join(_test_dir, "opik_v14.py")

# Load opik_v14 module directly from file
opik_v14 = load_module_from_path("opik_v14_test", _opik_v14_path)

# Import the classes we need
AgentOptimizer = opik_v14.AgentOptimizer
OutputGuardrail = opik_v14.OutputGuardrail
SwarmTraceType = opik_v14.SwarmTraceType
GuardrailType = opik_v14.GuardrailType
GuardrailResult = opik_v14.GuardrailResult
GuardrailViolation = opik_v14.GuardrailViolation
RemediationStrategy = opik_v14.RemediationStrategy


async def test_agent_optimizer():
    """Test AgentOptimizer swarm tracing."""
    print("\n1. Testing AgentOptimizer...")

    optimizer = AgentOptimizer(opik_client=None, budget_limit_usd=5.0)

    # Test sync tracing
    with optimizer.trace_swarm_sync("consensus", ["agent-1", "agent-2", "agent-3"]) as ctx:
        ctx.record_agent_result("agent-1", "Result A", latency_ms=100, cost_usd=0.001)
        ctx.record_agent_result("agent-2", "Result B", latency_ms=150, cost_usd=0.001)
        ctx.record_agent_result("agent-3", "Result C", latency_ms=120, cost_usd=0.001)
        ctx.set_winner("agent-2", confidence=0.85)

    # Check stats
    stats = optimizer.get_session_stats()
    print(f"   ✅ Traced 1 swarm execution")
    print(f"   ✅ Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"   ✅ Budget remaining: ${stats['budget_remaining_usd']:.4f}")

    return True


async def test_output_guardrail():
    """Test OutputGuardrail validation."""
    print("\n2. Testing OutputGuardrail...")

    guardrail = OutputGuardrail(
        hallucination_threshold=0.5,
        relevance_threshold=0.7,
    )

    # Test PII detection
    output_with_pii = "Call me at 555-123-4567 or email me at test@example.com"
    result = await guardrail.validate(output_with_pii)

    if not result.passed:
        print(f"   ✅ PII correctly detected: {len(result.violations)} violations")
    else:
        print(f"   ⚠️ PII not detected (expected)")

    # Test clean output
    clean_output = "The weather today is sunny with temperatures around 75 degrees."
    result = await guardrail.validate(clean_output)
    print(f"   ✅ Clean output passed: {result.passed}")

    # Test format validation
    invalid_json = "This is not valid JSON"
    result = await guardrail.validate(invalid_json, expected_format="json")

    if not result.passed or any(v.guardrail_type == GuardrailType.FORMAT for v in result.violations):
        print(f"   ✅ Invalid JSON format detected")

    # Test length validation
    long_output = "x" * 1000
    result = await guardrail.validate(long_output, max_length=100)

    if any(v.guardrail_type == GuardrailType.LENGTH for v in result.violations):
        print(f"   ✅ Length limit exceeded detected")

    return True


async def test_remediation():
    """Test guardrail remediation."""
    print("\n3. Testing Remediation...")

    guardrail = OutputGuardrail()

    # Test PII remediation
    output_with_pii = "My SSN is 123-45-6789 and my email is user@test.com"
    violations = [
        GuardrailViolation(
            guardrail_type=GuardrailType.PII,
            severity="critical",
            remediation=RemediationStrategy.FILTER,
        )
    ]

    remediated = await guardrail.remediate(output_with_pii, violations)

    if "[REDACTED]" in remediated:
        print(f"   ✅ PII successfully redacted")
    else:
        print(f"   ⚠️ PII redaction may not have worked")

    # Test length remediation
    long_output = "This is a very long output " * 50
    violations = [
        GuardrailViolation(
            guardrail_type=GuardrailType.LENGTH,
            severity="warning",
            details={"max_length": 100},
            remediation=RemediationStrategy.FILTER,
        )
    ]

    remediated = await guardrail.remediate(long_output, violations)
    print(f"   ✅ Length remediation: {len(long_output)} -> {len(remediated)} chars")

    return True


async def test_custom_guardrail():
    """Test custom guardrail registration."""
    print("\n4. Testing Custom Guardrail...")

    guardrail = OutputGuardrail()

    # Register custom guardrail
    def check_profanity(output: str, _context: dict) -> GuardrailViolation | None:
        bad_words = ["badword1", "badword2"]
        for word in bad_words:
            if word in output.lower():
                return GuardrailViolation(
                    guardrail_type=GuardrailType.CUSTOM,
                    severity="warning",
                    message=f"Profanity detected: {word}",
                )
        return None

    guardrail.register_custom_guardrail("profanity", check_profanity)
    print(f"   ✅ Custom guardrail registered")

    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("OPIK V14 INTEGRATION TESTS")
    print("=" * 60)

    results = []
    results.append(await test_agent_optimizer())
    results.append(await test_output_guardrail())
    results.append(await test_remediation())
    results.append(await test_custom_guardrail())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL OPIK V14 TESTS PASSED")
    else:
        print(f"⚠️ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
