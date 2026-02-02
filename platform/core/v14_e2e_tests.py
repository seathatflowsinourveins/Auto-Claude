#!/usr/bin/env python3
"""
V14 Platform E2E Tests - Real API Verification

Tests:
1. Sleep-time agent enablement (verify all 3 agents have it enabled)
2. Message round-trip (send message to UNLEASH agent, verify response)
3. Memory persistence (create passage, search it)
4. Parallel health check performance
5. Cross-session memory (verify blocks persist)

Usage:
    python v14_e2e_tests.py

All tests use REAL Letta Cloud API - no mocks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Verify environment
if not os.environ.get("LETTA_API_KEY"):
    print("ERROR: LETTA_API_KEY not set")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class V14E2ETests:
    """E2E tests for V14 optimizations using real Letta Cloud API."""

    # Verified Cloud Agent IDs
    AGENTS = {
        "unleash": "agent-daee71d2-193b-485e-bda4-ee44752635fe",
        "witness": "agent-bbcc0a74-5ff8-4ccd-83bc-b7282c952589",
        "alphaforge": "agent-5676da61-c57c-426e-a0f6-390fd9dfcf94",
    }

    def __init__(self):
        from letta_client import Letta
        self.client = Letta(
            api_key=os.environ["LETTA_API_KEY"],
            base_url="https://api.letta.com"
        )
        self.results: list[TestResult] = []

    def test_sleeptime_enabled(self) -> TestResult:
        """Test 1: Verify sleep-time is enabled on all agents."""
        start = time.perf_counter()
        try:
            enabled_count = 0
            agent_status = {}

            for name, agent_id in self.AGENTS.items():
                agent = self.client.agents.retrieve(agent_id)
                enabled = getattr(agent, 'enable_sleeptime', False) or False
                # V116: Fixed attribute name from 'managed_group' to 'multi_agent_group'
                multi_agent_group = getattr(agent, 'multi_agent_group', None)

                agent_status[name] = {
                    "enabled": enabled,
                    "managed_group_id": multi_agent_group.id if multi_agent_group else None
                }

                if enabled:
                    enabled_count += 1

            duration = (time.perf_counter() - start) * 1000
            passed = enabled_count == 3

            return TestResult(
                name="sleeptime_enabled",
                passed=passed,
                duration_ms=duration,
                message=f"{enabled_count}/3 agents have sleep-time enabled",
                details=agent_status
            )

        except Exception as e:
            return TestResult(
                name="sleeptime_enabled",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                message=f"Error: {str(e)}"
            )

    def test_message_roundtrip(self) -> TestResult:
        """Test 2: Send message to UNLEASH agent, verify response."""
        start = time.perf_counter()
        agent_id = self.AGENTS["unleash"]

        try:
            # Send a simple test message
            response = self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{
                    "role": "user",
                    "content": "V14 E2E Test: Respond with 'OK' if you received this message."
                }]
            )

            duration = (time.perf_counter() - start) * 1000

            # Check for response
            assistant_messages = [
                m for m in response.messages
                if hasattr(m, 'message_type') and m.message_type == "assistant_message"
            ]

            passed = len(assistant_messages) > 0
            content = assistant_messages[0].content if assistant_messages else "No response"

            return TestResult(
                name="message_roundtrip",
                passed=passed,
                duration_ms=duration,
                message=f"Agent responded: {content[:100]}..." if len(content) > 100 else f"Agent responded: {content}",
                details={
                    "message_count": len(response.messages),
                    "has_assistant_response": passed
                }
            )

        except Exception as e:
            return TestResult(
                name="message_roundtrip",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                message=f"Error: {str(e)}"
            )

    def test_memory_persistence(self) -> TestResult:
        """Test 3: Create passage, search it, verify persistence."""
        start = time.perf_counter()
        agent_id = self.AGENTS["unleash"]
        test_content = f"V14_E2E_TEST_PASSAGE_{int(time.time())}"

        try:
            # Create a passage
            created = self.client.agents.passages.create(
                agent_id=agent_id,
                text=test_content,
                tags=["v14_test", "e2e"]
            )

            # passages.create returns a list
            passage_id = created[0].id if isinstance(created, list) else created.id

            # Search for it
            search_result = self.client.agents.passages.search(
                agent_id=agent_id,
                query=test_content,
                top_k=5
            )

            # Check if found
            found = False
            for result in search_result.results:
                if test_content in result.content:
                    found = True
                    break

            # Cleanup - delete the test passage
            try:
                self.client.agents.passages.delete(
                    memory_id=passage_id,
                    agent_id=agent_id
                )
            except Exception:
                pass  # Cleanup failure is not critical

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="memory_persistence",
                passed=found,
                duration_ms=duration,
                message="Passage created and found via search" if found else "Passage not found after creation",
                details={
                    "passage_id": passage_id,
                    "search_results_count": len(search_result.results)
                }
            )

        except Exception as e:
            return TestResult(
                name="memory_persistence",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                message=f"Error: {str(e)}"
            )

    def test_blocks_access(self) -> TestResult:
        """Test 4: Verify memory blocks are accessible."""
        start = time.perf_counter()
        agent_id = self.AGENTS["unleash"]

        try:
            # List blocks
            blocks = list(self.client.agents.blocks.list(agent_id))

            # Check for required blocks
            block_labels = [b.label for b in blocks]
            has_human = "human" in block_labels
            has_persona = "persona" in block_labels

            duration = (time.perf_counter() - start) * 1000
            passed = has_human and has_persona

            return TestResult(
                name="blocks_access",
                passed=passed,
                duration_ms=duration,
                message=f"Found blocks: {block_labels}",
                details={
                    "block_count": len(blocks),
                    "has_human": has_human,
                    "has_persona": has_persona,
                    "labels": block_labels
                }
            )

        except Exception as e:
            return TestResult(
                name="blocks_access",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                message=f"Error: {str(e)}"
            )

    async def test_parallel_health_performance(self) -> TestResult:
        """Test 5: Verify parallel health checks are faster than sequential."""
        import httpx

        async def check_letta_cloud():
            start = time.perf_counter()
            try:
                # Use the sync client in executor
                agents = list(self.client.agents.list(limit=1))
                return (time.perf_counter() - start) * 1000, True
            except Exception:
                return (time.perf_counter() - start) * 1000, False

        async def check_endpoint(url: str):
            start = time.perf_counter()
            try:
                async with httpx.AsyncClient() as client:
                    await client.get(url, timeout=3.0)
                return (time.perf_counter() - start) * 1000, True
            except Exception:
                return (time.perf_counter() - start) * 1000, False

        start = time.perf_counter()

        try:
            # Run checks in parallel
            results = await asyncio.gather(
                check_letta_cloud(),
                check_endpoint("http://localhost:6333"),  # Qdrant
                check_endpoint("http://localhost:8283"),  # Letta local
                return_exceptions=True
            )

            total_parallel = (time.perf_counter() - start) * 1000

            # Calculate what sequential would be
            individual_times = [
                r[0] if isinstance(r, tuple) else 3000  # 3s timeout
                for r in results
            ]
            estimated_sequential = sum(individual_times)

            speedup = estimated_sequential / total_parallel if total_parallel > 0 else 1

            return TestResult(
                name="parallel_health_performance",
                passed=speedup > 1.5,  # At least 1.5x faster
                duration_ms=total_parallel,
                message=f"Parallel: {total_parallel:.0f}ms vs Sequential estimate: {estimated_sequential:.0f}ms ({speedup:.1f}x speedup)",
                details={
                    "parallel_ms": total_parallel,
                    "sequential_estimate_ms": estimated_sequential,
                    "speedup_factor": speedup
                }
            )

        except Exception as e:
            return TestResult(
                name="parallel_health_performance",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                message=f"Error: {str(e)}"
            )

    def run_all_tests(self) -> list[TestResult]:
        """Run all E2E tests."""
        print("\n" + "=" * 60)
        print("V14 PLATFORM E2E TESTS - REAL API VERIFICATION")
        print("=" * 60 + "\n")

        # Sync tests
        tests = [
            ("Sleep-time Enabled", self.test_sleeptime_enabled),
            ("Message Round-trip", self.test_message_roundtrip),
            ("Memory Persistence", self.test_memory_persistence),
            ("Blocks Access", self.test_blocks_access),
        ]

        for name, test_fn in tests:
            print(f"Running: {name}...", end=" ", flush=True)
            result = test_fn()
            self.results.append(result)
            icon = "✅" if result.passed else "❌"
            print(f"{icon} ({result.duration_ms:.0f}ms)")
            if not result.passed:
                print(f"   └─ {result.message}")

        # Async test
        print("Running: Parallel Health Performance...", end=" ", flush=True)
        result = asyncio.run(self.test_parallel_health_performance())
        self.results.append(result)
        icon = "✅" if result.passed else "❌"
        print(f"{icon} ({result.duration_ms:.0f}ms)")
        if result.message:
            print(f"   └─ {result.message}")

        return self.results

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            icon = "✅" if r.passed else "❌"
            print(f"{icon} {r.name}: {r.message}")

        print("-" * 60)
        print(f"TOTAL: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 ALL TESTS PASSED - V14 OPTIMIZATIONS VERIFIED")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")

        return passed == total


if __name__ == "__main__":
    tester = V14E2ETests()
    tester.run_all_tests()
    success = tester.print_summary()
    sys.exit(0 if success else 1)
