"""
Everything Claude Code Pattern Integration Tests - ITERATION 8.

Tests verify integration of ECC patterns with orchestration layer:
- Opik tracing integration for observability
- Verification loop patterns for result validation
- Agent patterns for specialized sub-agents
- TDD workflow patterns
- Security review patterns

These tests validate L5 (Observability) integration with L1 (Orchestration).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest


# =============================================================================
# Opik Availability Check
# =============================================================================

def check_opik_available() -> bool:
    """Check if Opik is available for tracing."""
    try:
        import opik
        return True
    except ImportError:
        return False


OPIK_AVAILABLE = check_opik_available()


# =============================================================================
# Pattern Dataclasses
# =============================================================================

@dataclass
class VerificationResult:
    """Result from a verification check."""
    passed: bool
    checks_run: int
    checks_passed: int
    checks_failed: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task definition for specialized agents."""
    name: str
    description: str
    agent_type: str  # planner, code-reviewer, security-reviewer, etc.
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class TDDStep:
    """Step in the TDD workflow."""
    phase: str  # red, green, refactor
    description: str
    tests_written: int = 0
    tests_passing: int = 0
    coverage: float = 0.0


# =============================================================================
# Verification Loop Implementation
# =============================================================================

class VerificationLoop:
    """
    Implements the verification loop pattern from Everything Claude Code.

    Runs a series of verification checks on orchestration results
    to ensure correctness, completeness, and quality.
    """

    def __init__(self) -> None:
        self.checks: list[tuple[str, Callable]] = []
        self.results: list[VerificationResult] = []

    def add_check(self, name: str, check_fn: Callable[[Any], bool]) -> None:
        """Add a verification check."""
        self.checks.append((name, check_fn))

    def verify(self, target: Any) -> VerificationResult:
        """Run all verification checks on the target."""
        failures = []
        checks_passed = 0

        for name, check_fn in self.checks:
            try:
                if check_fn(target):
                    checks_passed += 1
                else:
                    failures.append({"check": name, "error": "Check returned False"})
            except Exception as e:
                failures.append({"check": name, "error": str(e)})

        result = VerificationResult(
            passed=len(failures) == 0,
            checks_run=len(self.checks),
            checks_passed=checks_passed,
            checks_failed=len(failures),
            failures=failures,
        )
        self.results.append(result)
        return result

    def get_success_rate(self) -> float:
        """Get the overall success rate across all verifications."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passed)
        return passed / len(self.results)


# =============================================================================
# TDD Workflow Implementation
# =============================================================================

class TDDWorkflow:
    """
    Implements the TDD workflow pattern from Everything Claude Code.

    Ensures test-driven development with:
    - RED: Write failing tests first
    - GREEN: Write minimal code to pass
    - REFACTOR: Improve while keeping tests green
    """

    def __init__(self) -> None:
        self.steps: list[TDDStep] = []
        self.current_phase: str = "red"
        self.target_coverage: float = 0.8

    def add_red_step(self, description: str, tests_written: int) -> TDDStep:
        """Record a RED phase step (writing failing tests)."""
        step = TDDStep(
            phase="red",
            description=description,
            tests_written=tests_written,
            tests_passing=0,
        )
        self.steps.append(step)
        self.current_phase = "green"
        return step

    def add_green_step(self, description: str, tests_passing: int) -> TDDStep:
        """Record a GREEN phase step (making tests pass)."""
        step = TDDStep(
            phase="green",
            description=description,
            tests_written=self.steps[-1].tests_written if self.steps else 0,
            tests_passing=tests_passing,
        )
        self.steps.append(step)
        self.current_phase = "refactor"
        return step

    def add_refactor_step(self, description: str, coverage: float) -> TDDStep:
        """Record a REFACTOR phase step (improving code quality)."""
        step = TDDStep(
            phase="refactor",
            description=description,
            tests_written=self.steps[-1].tests_written if self.steps else 0,
            tests_passing=self.steps[-1].tests_passing if self.steps else 0,
            coverage=coverage,
        )
        self.steps.append(step)
        self.current_phase = "red"  # Ready for next cycle
        return step

    def is_coverage_met(self) -> bool:
        """Check if target coverage is met."""
        if not self.steps:
            return False
        latest_coverage = next(
            (s.coverage for s in reversed(self.steps) if s.coverage > 0),
            0.0
        )
        return latest_coverage >= self.target_coverage

    def get_workflow_summary(self) -> dict[str, Any]:
        """Get a summary of the TDD workflow."""
        return {
            "total_steps": len(self.steps),
            "red_steps": sum(1 for s in self.steps if s.phase == "red"),
            "green_steps": sum(1 for s in self.steps if s.phase == "green"),
            "refactor_steps": sum(1 for s in self.steps if s.phase == "refactor"),
            "current_phase": self.current_phase,
            "coverage_met": self.is_coverage_met(),
        }


# =============================================================================
# Agent Manager for ECC Agents
# =============================================================================

class AgentManager:
    """
    Manages specialized agents from Everything Claude Code.

    Available agent types:
    - planner: Creates implementation plans
    - code-reviewer: Reviews code quality
    - security-reviewer: Security analysis
    - tdd-guide: Test-driven development guidance
    - architect: System architecture design
    """

    AGENT_TYPES = {
        "planner": "Creates step-by-step implementation plans",
        "code-reviewer": "Reviews code for quality, patterns, bugs",
        "security-reviewer": "Analyzes code for security vulnerabilities",
        "tdd-guide": "Guides test-driven development workflow",
        "architect": "Designs system architecture",
        "e2e-runner": "Runs end-to-end tests",
        "build-error-resolver": "Resolves build errors",
        "refactor-cleaner": "Cleans and refactors code",
        "doc-updater": "Updates documentation",
    }

    def __init__(self) -> None:
        self.tasks: list[AgentTask] = []
        self.completed_tasks: list[AgentTask] = []

    def create_task(
        self,
        name: str,
        description: str,
        agent_type: str,
        inputs: dict[str, Any] | None = None,
    ) -> AgentTask:
        """Create a new agent task."""
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}")

        task = AgentTask(
            name=name,
            description=description,
            agent_type=agent_type,
            inputs=inputs or {},
        )
        self.tasks.append(task)
        return task

    def run_task(self, task: AgentTask) -> dict[str, Any]:
        """
        Simulate running an agent task.

        In production, this would delegate to actual agent implementations.
        """
        task.status = "running"

        # Simulate agent execution
        time.sleep(0.01)  # Minimal delay

        task.status = "completed"
        task.outputs = {
            "result": f"Completed {task.agent_type} task: {task.name}",
            "agent": task.agent_type,
            "success": True,
        }

        self.completed_tasks.append(task)
        return task.outputs

    def get_agent_types(self) -> dict[str, str]:
        """Get available agent types and descriptions."""
        return self.AGENT_TYPES.copy()


# =============================================================================
# Tests: Verification Loop Pattern
# =============================================================================

class TestVerificationLoop:
    """Test the verification loop pattern."""

    def test_verification_loop_creation(self):
        """Test creating a verification loop."""
        loop = VerificationLoop()
        assert len(loop.checks) == 0
        assert len(loop.results) == 0

    def test_add_verification_checks(self):
        """Test adding verification checks."""
        loop = VerificationLoop()

        loop.add_check("not_empty", lambda x: len(x) > 0)
        loop.add_check("has_key", lambda x: "key" in x)

        assert len(loop.checks) == 2

    def test_verify_passing_target(self):
        """Test verification with passing target."""
        loop = VerificationLoop()
        loop.add_check("has_name", lambda x: "name" in x)
        loop.add_check("name_not_empty", lambda x: len(x.get("name", "")) > 0)

        target = {"name": "test_item"}
        result = loop.verify(target)

        assert result.passed is True
        assert result.checks_run == 2
        assert result.checks_passed == 2
        assert result.checks_failed == 0

    def test_verify_failing_target(self):
        """Test verification with failing target."""
        loop = VerificationLoop()
        loop.add_check("has_name", lambda x: "name" in x)
        loop.add_check("has_id", lambda x: "id" in x)

        target = {"name": "test"}  # Missing id
        result = loop.verify(target)

        assert result.passed is False
        assert result.checks_run == 2
        assert result.checks_passed == 1
        assert result.checks_failed == 1

    def test_success_rate_calculation(self):
        """Test success rate across multiple verifications."""
        loop = VerificationLoop()
        loop.add_check("has_value", lambda x: x.get("value", 0) > 0)

        # Run multiple verifications
        loop.verify({"value": 10})  # Pass
        loop.verify({"value": 0})   # Fail
        loop.verify({"value": 5})   # Pass
        loop.verify({})             # Fail

        rate = loop.get_success_rate()
        assert rate == 0.5  # 2 out of 4 passed


# =============================================================================
# Tests: TDD Workflow Pattern
# =============================================================================

class TestTDDWorkflow:
    """Test the TDD workflow pattern."""

    def test_workflow_creation(self):
        """Test creating a TDD workflow."""
        workflow = TDDWorkflow()
        assert workflow.current_phase == "red"
        assert workflow.target_coverage == 0.8
        assert len(workflow.steps) == 0

    def test_red_phase(self):
        """Test adding a RED phase step."""
        workflow = TDDWorkflow()
        step = workflow.add_red_step("Write test for new feature", 3)

        assert step.phase == "red"
        assert step.tests_written == 3
        assert step.tests_passing == 0
        assert workflow.current_phase == "green"

    def test_green_phase(self):
        """Test adding a GREEN phase step."""
        workflow = TDDWorkflow()
        workflow.add_red_step("Write failing tests", 3)
        step = workflow.add_green_step("Implement feature", 3)

        assert step.phase == "green"
        assert step.tests_passing == 3
        assert workflow.current_phase == "refactor"

    def test_refactor_phase(self):
        """Test adding a REFACTOR phase step."""
        workflow = TDDWorkflow()
        workflow.add_red_step("Write tests", 5)
        workflow.add_green_step("Implement", 5)
        step = workflow.add_refactor_step("Clean up code", 0.85)

        assert step.phase == "refactor"
        assert step.coverage == 0.85
        assert workflow.current_phase == "red"  # Ready for next cycle

    def test_coverage_check(self):
        """Test coverage threshold checking."""
        workflow = TDDWorkflow()
        workflow.target_coverage = 0.8

        # Before any steps
        assert workflow.is_coverage_met() is False

        # Add steps with insufficient coverage
        workflow.add_red_step("Tests", 5)
        workflow.add_green_step("Implement", 5)
        workflow.add_refactor_step("Refactor", 0.7)
        assert workflow.is_coverage_met() is False

        # Add step meeting coverage
        workflow.add_refactor_step("More refactoring", 0.85)
        assert workflow.is_coverage_met() is True

    def test_workflow_summary(self):
        """Test getting workflow summary."""
        workflow = TDDWorkflow()
        workflow.add_red_step("Write tests", 3)
        workflow.add_green_step("Implement", 3)
        workflow.add_refactor_step("Clean up", 0.82)

        summary = workflow.get_workflow_summary()

        assert summary["total_steps"] == 3
        assert summary["red_steps"] == 1
        assert summary["green_steps"] == 1
        assert summary["refactor_steps"] == 1
        assert summary["coverage_met"] is True


# =============================================================================
# Tests: Agent Manager Pattern
# =============================================================================

class TestAgentManager:
    """Test the agent manager pattern."""

    def test_agent_manager_creation(self):
        """Test creating an agent manager."""
        manager = AgentManager()
        assert len(manager.tasks) == 0
        assert len(manager.completed_tasks) == 0

    def test_get_agent_types(self):
        """Test getting available agent types."""
        manager = AgentManager()
        types = manager.get_agent_types()

        assert "planner" in types
        assert "code-reviewer" in types
        assert "security-reviewer" in types
        assert "tdd-guide" in types
        assert len(types) == 9

    def test_create_task(self):
        """Test creating an agent task."""
        manager = AgentManager()
        task = manager.create_task(
            name="Review auth module",
            description="Security review of authentication",
            agent_type="security-reviewer",
            inputs={"files": ["auth.py", "login.py"]},
        )

        assert task.name == "Review auth module"
        assert task.agent_type == "security-reviewer"
        assert task.status == "pending"
        assert len(manager.tasks) == 1

    def test_create_task_invalid_type(self):
        """Test creating task with invalid agent type."""
        manager = AgentManager()

        with pytest.raises(ValueError, match="Unknown agent type"):
            manager.create_task(
                name="Invalid task",
                description="This should fail",
                agent_type="nonexistent-agent",
            )

    def test_run_task(self):
        """Test running an agent task."""
        manager = AgentManager()
        task = manager.create_task(
            name="Plan feature",
            description="Create implementation plan",
            agent_type="planner",
        )

        result = manager.run_task(task)

        assert task.status == "completed"
        assert "success" in result
        assert result["success"] is True
        assert len(manager.completed_tasks) == 1


# =============================================================================
# Tests: Opik Integration (Conditional)
# =============================================================================

@pytest.mark.skipif(not OPIK_AVAILABLE, reason="Opik not installed")
class TestOpikIntegration:
    """Test Opik tracing integration."""

    def test_opik_import(self):
        """Test Opik can be imported."""
        import opik
        assert hasattr(opik, "track")

    def test_opik_tracing_decorator(self):
        """Test Opik tracing decorator exists."""
        import opik

        # Should not raise
        @opik.track(name="test_function")
        def sample_function():
            return "result"

        assert sample_function() == "result"


# =============================================================================
# Tests: Pattern Integration with Orchestration
# =============================================================================

class TestPatternOrchestrationIntegration:
    """Test integration of ECC patterns with orchestration layer."""

    @pytest.mark.asyncio
    async def test_verification_with_orchestration_result(self):
        """Test verification loop with OrchestrationResult."""
        from core.orchestration import OrchestrationResult

        # Create an orchestration result
        result = OrchestrationResult(
            success=True,
            framework="claude_agent_sdk",
            output={"message": "Task completed"},
            metadata={"duration": 1.5},
        )

        # Set up verification loop
        loop = VerificationLoop()
        loop.add_check("is_success", lambda r: r.success is True)
        loop.add_check("has_framework", lambda r: len(r.framework) > 0)
        loop.add_check("has_output", lambda r: r.output is not None)
        loop.add_check("no_error", lambda r: r.error is None)

        # Verify the result
        verification = loop.verify(result)

        assert verification.passed is True
        assert verification.checks_passed == 4

    @pytest.mark.asyncio
    async def test_agent_task_with_embedding_layer(self):
        """Test agent tasks that use embedding layer."""
        from core.orchestration.embedding_layer import create_embedding_layer

        # Create embedding layer
        layer = create_embedding_layer(model="voyage-3")

        # Create agent task for semantic search
        manager = AgentManager()
        task = manager.create_task(
            name="Semantic search setup",
            description="Configure embedding layer for search",
            agent_type="architect",
            inputs={
                "model": layer.config.model,
                "dimension": 1024,
                "cache_enabled": layer.config.cache_enabled,
            },
        )

        result = manager.run_task(task)

        assert result["success"] is True
        assert task.inputs["model"] == "voyage-3"

    @pytest.mark.asyncio
    async def test_tdd_workflow_with_agent_sdk(self):
        """Test TDD workflow with Agent SDK layer."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        # TDD Workflow for agent development
        workflow = TDDWorkflow()

        # RED: Write failing tests for agent config
        workflow.add_red_step("Write tests for AgentConfig validation", 3)

        # GREEN: Create agent with valid config
        config = AgentConfig(
            name="tdd_agent",
            tools=["Read", "Write", "Bash"],
        )
        agent = Agent(config)

        workflow.add_green_step("Implement AgentConfig with tools", 3)

        # REFACTOR: Verify tool building
        tools = agent._build_tools()
        assert len(tools) == 3

        workflow.add_refactor_step("Verified tool building", 0.85)

        summary = workflow.get_workflow_summary()
        assert summary["coverage_met"] is True


# =============================================================================
# Tests: Performance of Pattern Operations
# =============================================================================

class TestPatternPerformance:
    """Test performance of ECC pattern operations."""

    def test_verification_loop_speed(self):
        """Test verification loop is fast."""
        loop = VerificationLoop()

        # Add multiple checks
        for i in range(10):
            loop.add_check(f"check_{i}", lambda x, i=i: True)

        start = time.time()
        for _ in range(100):
            loop.verify({"data": "test"})
        duration = time.time() - start

        # 100 verifications with 10 checks each should be < 0.5s
        assert duration < 0.5, f"Verification too slow: {duration}s"

    def test_agent_manager_speed(self):
        """Test agent manager is fast."""
        manager = AgentManager()

        start = time.time()
        for i in range(50):
            task = manager.create_task(
                name=f"Task {i}",
                description="Test task",
                agent_type="planner",
            )
            manager.run_task(task)
        duration = time.time() - start

        # 50 task creations + runs should be < 2s
        assert duration < 2.0, f"Agent manager too slow: {duration}s"

    def test_tdd_workflow_speed(self):
        """Test TDD workflow tracking is fast."""
        start = time.time()
        for _ in range(100):
            workflow = TDDWorkflow()
            workflow.add_red_step("Write tests", 5)
            workflow.add_green_step("Implement", 5)
            workflow.add_refactor_step("Clean up", 0.82)
            workflow.get_workflow_summary()
        duration = time.time() - start

        # 100 complete TDD cycles should be < 0.1s
        assert duration < 0.1, f"TDD workflow too slow: {duration}s"


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
