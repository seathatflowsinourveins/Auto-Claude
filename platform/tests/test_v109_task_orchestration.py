#!/usr/bin/env python3
"""
Test Suite for V10.9 Task Orchestration Patterns

Tests for:
- Task-Master PRD Patterns (PRDTask, PRDWorkflow, ComplexityAnalysis, etc.)
- Shrimp Chain-of-Thought Patterns (ChainOfThought, ThoughtProcess, StructuredWorkflow)
- Semgrep Security Patterns (SecurityScanner, SecurityAudit, SemgrepRule)

Based on official documentation:
- claude-task-master v1.15+: https://github.com/eyaltoledano/claude-task-master
- mcp-shrimp-task-manager v1.1+: https://github.com/cjo4m06/mcp-shrimp-task-manager
- Semgrep MCP v1.0+: https://github.com/semgrep/mcp
"""

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # Task-Master PRD Patterns
    TaskPriority,
    ComplexityLevel,
    ToolMode,
    PRDTask,
    ComplexityAnalysis,
    TaggedTaskList,
    TaskDependencyGraph,
    PRDWorkflow,
    # Shrimp Chain-of-Thought Patterns
    TaskWorkflowMode,
    ChainOfThought,
    ThoughtProcess,
    TaskDependency,
    PersistentTaskMemory,
    StructuredWorkflow,
    # Semgrep Security Patterns
    SecuritySeverity,
    RuleCategory,
    SecurityFinding,
    ASTNode,
    SemgrepRule,
    ASTAnalyzer,
    SecurityScanner,
    SecurityAudit,
)


# =============================================================================
# Task-Master PRD Pattern Tests
# =============================================================================


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert TaskPriority.CRITICAL.value == "critical"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.DEFERRED.value == "deferred"

    def test_priority_from_string(self):
        """Test creating priority from string value."""
        assert TaskPriority("critical") == TaskPriority.CRITICAL
        assert TaskPriority("high") == TaskPriority.HIGH


class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert ComplexityLevel.TRIVIAL.value == "trivial"
        assert ComplexityLevel.SIMPLE.value == "simple"
        assert ComplexityLevel.MODERATE.value == "moderate"
        assert ComplexityLevel.COMPLEX.value == "complex"
        assert ComplexityLevel.EPIC.value == "epic"


class TestToolMode:
    """Tests for ToolMode enum."""

    def test_tool_mode_values(self):
        """Test tool mode enum values."""
        assert ToolMode.CORE.value == "core"
        assert ToolMode.STANDARD.value == "standard"
        assert ToolMode.ALL.value == "all"


class TestPRDTask:
    """Tests for PRDTask dataclass."""

    def test_task_creation(self):
        """Test creating a PRD task."""
        task = PRDTask(
            id="task_1",
            title="Implement authentication",
            description="Add user authentication system"
        )
        assert task.id == "task_1"
        assert task.title == "Implement authentication"
        assert task.status == "pending"
        assert task.priority == TaskPriority.MEDIUM

    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        task = PRDTask(
            id="task_2",
            title="Add login form",
            description="Create login UI",
            dependencies=["task_1"]
        )
        assert task.dependencies == ["task_1"]

    def test_task_to_dict(self):
        """Test task serialization."""
        task = PRDTask(
            id="task_1",
            title="Test Task",
            description="Description"
        )
        data = task.to_dict()
        assert data["id"] == "task_1"
        assert data["title"] == "Test Task"
        assert data["priority"] == "medium"
        assert data["complexity"] == "moderate"

    def test_task_from_dict(self):
        """Test task deserialization."""
        data = {
            "id": "task_1",
            "title": "Test Task",
            "description": "Description",
            "priority": "high",
            "complexity": "complex",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        task = PRDTask.from_dict(data)
        assert task.id == "task_1"
        assert task.priority == TaskPriority.HIGH
        assert task.complexity == ComplexityLevel.COMPLEX

    def test_task_with_subtasks(self):
        """Test task with subtasks."""
        parent = PRDTask(
            id="task_1",
            title="Parent Task",
            description="Parent",
            subtasks=[
                PRDTask(id="task_1_1", title="Subtask 1", description="Sub 1"),
                PRDTask(id="task_1_2", title="Subtask 2", description="Sub 2")
            ]
        )
        assert len(parent.subtasks) == 2
        assert parent.subtasks[0].id == "task_1_1"


class TestComplexityAnalysis:
    """Tests for ComplexityAnalysis dataclass."""

    def test_complexity_analysis_creation(self):
        """Test creating complexity analysis."""
        analysis = ComplexityAnalysis(
            task_id="task_1",
            level=ComplexityLevel.EPIC,
            score=0.9,
            requires_decomposition=True,
            suggested_subtask_count=5
        )
        assert analysis.task_id == "task_1"
        assert analysis.level == ComplexityLevel.EPIC
        assert analysis.requires_decomposition is True

    def test_complexity_analysis_to_dict(self):
        """Test analysis serialization."""
        analysis = ComplexityAnalysis(
            task_id="task_1",
            level=ComplexityLevel.MODERATE,
            score=0.5
        )
        data = analysis.to_dict()
        assert data["level"] == "moderate"
        assert data["score"] == 0.5


class TestTaggedTaskList:
    """Tests for TaggedTaskList."""

    def test_tagged_list_creation(self):
        """Test creating a tagged task list."""
        task_list = TaggedTaskList(tag="feature-auth")
        assert task_list.tag == "feature-auth"
        assert len(task_list.tasks) == 0

    def test_add_task_to_list(self):
        """Test adding tasks to list."""
        task_list = TaggedTaskList(tag="feature-auth")
        task = PRDTask(id="task_1", title="Test", description="Desc")
        task_list.add_task(task)

        assert len(task_list.tasks) == 1
        assert "feature-auth" in task.tags

    def test_get_next_task(self):
        """Test getting next executable task."""
        task_list = TaggedTaskList(tag="test")

        # Add tasks with dependency
        task1 = PRDTask(id="task_1", title="First", description="Desc")
        task2 = PRDTask(id="task_2", title="Second", description="Desc", dependencies=["task_1"])

        task_list.add_task(task1)
        task_list.add_task(task2)

        # First task should be next (no deps)
        next_task = task_list.get_next_task()
        assert next_task.id == "task_1"

    def test_get_tasks_by_status(self):
        """Test filtering tasks by status."""
        task_list = TaggedTaskList(tag="test")
        task1 = PRDTask(id="task_1", title="Test 1", description="Desc", status="pending")
        task2 = PRDTask(id="task_2", title="Test 2", description="Desc", status="completed")

        task_list.add_task(task1)
        task_list.add_task(task2)

        pending = task_list.get_tasks_by_status("pending")
        assert len(pending) == 1
        assert pending[0].id == "task_1"


class TestTaskDependencyGraph:
    """Tests for TaskDependencyGraph."""

    def test_graph_creation(self):
        """Test creating dependency graph."""
        graph = TaskDependencyGraph()
        assert len(graph._tasks) == 0

    def test_add_tasks_to_graph(self):
        """Test adding tasks to graph."""
        graph = TaskDependencyGraph()
        task1 = PRDTask(id="task_1", title="First", description="Desc")
        task2 = PRDTask(id="task_2", title="Second", description="Desc", dependencies=["task_1"])

        graph.add_task(task1)
        graph.add_task(task2)

        assert len(graph._tasks) == 2

    def test_execution_order(self):
        """Test topological sort for execution order."""
        graph = TaskDependencyGraph()

        task1 = PRDTask(id="task_1", title="First", description="Desc")
        task2 = PRDTask(id="task_2", title="Second", description="Desc", dependencies=["task_1"])
        task3 = PRDTask(id="task_3", title="Third", description="Desc", dependencies=["task_2"])

        graph.add_tasks([task1, task2, task3])

        order = graph.get_execution_order()
        # task_1 must come before task_2, which must come before task_3
        assert order.index("task_1") < order.index("task_2")
        assert order.index("task_2") < order.index("task_3")

    def test_get_ready_tasks(self):
        """Test getting tasks ready for execution."""
        graph = TaskDependencyGraph()

        task1 = PRDTask(id="task_1", title="First", description="Desc", status="completed")
        task2 = PRDTask(id="task_2", title="Second", description="Desc", dependencies=["task_1"])

        graph.add_tasks([task1, task2])

        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "task_2"

    def test_get_blocked_tasks(self):
        """Test getting blocked tasks."""
        graph = TaskDependencyGraph()

        task1 = PRDTask(id="task_1", title="First", description="Desc")
        task2 = PRDTask(id="task_2", title="Second", description="Desc", dependencies=["task_1"])

        graph.add_tasks([task1, task2])

        blocked = graph.get_blocked_tasks()
        assert "task_2" in blocked


class TestPRDWorkflow:
    """Tests for PRDWorkflow."""

    def test_workflow_creation(self):
        """Test creating PRD workflow."""
        workflow = PRDWorkflow()
        assert workflow.default_tag == "master"
        assert workflow.tool_mode == ToolMode.STANDARD

    def test_parse_prd(self):
        """Test parsing PRD content."""
        workflow = PRDWorkflow()
        prd_content = """
        1. Implement user authentication
        2. Create database schema
        3. Build API endpoints
        """

        tasks = workflow.parse_prd(prd_content)
        assert len(tasks) == 3
        assert "authentication" in tasks[0].title.lower()

    def test_get_next_task(self):
        """Test getting next task from workflow."""
        workflow = PRDWorkflow()
        task = PRDTask(id="task_1", title="Test", description="Desc")
        workflow.task_lists["master"].add_task(task)

        next_task = workflow.get_next_task()
        assert next_task.id == "task_1"

    def test_set_task_status(self):
        """Test setting task status."""
        workflow = PRDWorkflow()
        task = PRDTask(id="task_1", title="Test", description="Desc")
        workflow.task_lists["master"].add_task(task)

        result = workflow.set_task_status("task_1", "completed")
        assert result is True
        assert task.status == "completed"

    def test_expand_task(self):
        """Test expanding task into subtasks."""
        workflow = PRDWorkflow()
        task = PRDTask(id="task_1", title="Complex Task", description="Desc")
        workflow.task_lists["master"].add_task(task)

        subtasks = workflow.expand_task("task_1", num_subtasks=3)
        assert len(subtasks) == 3
        assert len(task.subtasks) == 3


# =============================================================================
# Shrimp Chain-of-Thought Pattern Tests
# =============================================================================


class TestTaskWorkflowMode:
    """Tests for TaskWorkflowMode enum."""

    def test_workflow_mode_values(self):
        """Test workflow mode enum values."""
        assert TaskWorkflowMode.PLANNING.value == "planning"
        assert TaskWorkflowMode.EXECUTION.value == "execution"
        assert TaskWorkflowMode.CONTINUOUS.value == "continuous"
        assert TaskWorkflowMode.RESEARCH.value == "research"
        assert TaskWorkflowMode.REFLECTION.value == "reflection"


class TestChainOfThought:
    """Tests for ChainOfThought."""

    def test_chain_creation(self):
        """Test creating chain of thought."""
        chain = ChainOfThought(id="chain_1")
        assert chain.id == "chain_1"
        assert len(chain.steps) == 0
        assert chain.mode == TaskWorkflowMode.PLANNING

    def test_add_step(self):
        """Test adding reasoning steps."""
        chain = ChainOfThought(id="chain_1")
        idx = chain.add_step(thought="Analyzing requirements")

        assert idx == 0
        assert len(chain.steps) == 1
        assert chain.steps[0]["thought"] == "Analyzing requirements"

    def test_update_step(self):
        """Test updating existing step."""
        chain = ChainOfThought(id="chain_1")
        chain.add_step(thought="Initial thought")

        result = chain.update_step(0, observation="Found relevant code")
        assert result is True
        assert chain.steps[0]["observation"] == "Found relevant code"

    def test_advance_step(self):
        """Test advancing to next step."""
        chain = ChainOfThought(id="chain_1")
        chain.add_step(thought="Step 1")
        chain.add_step(thought="Step 2")

        assert chain.current_step == 0
        chain.advance()
        assert chain.current_step == 1

    def test_to_prompt(self):
        """Test converting chain to prompt."""
        chain = ChainOfThought(id="chain_1")
        chain.add_step(thought="Analyzing the problem")
        chain.add_step(thought="Identifying solutions", action="Search codebase")

        prompt = chain.to_prompt()
        assert "Chain of Thought" in prompt
        assert "Analyzing the problem" in prompt
        assert "Search codebase" in prompt


class TestThoughtProcess:
    """Tests for ThoughtProcess."""

    def test_thought_process_creation(self):
        """Test creating thought process."""
        chain = ChainOfThought(id="chain_1")
        process = ThoughtProcess(chain=chain)

        assert process.max_steps == 10
        assert process.enable_reflection is True

    def test_process_thought(self):
        """Test processing a thought."""
        chain = ChainOfThought(id="chain_1")
        process = ThoughtProcess(chain=chain)

        result = process.process("Analyzing the codebase")

        assert result["step_index"] == 0
        assert result["thought"] == "Analyzing the codebase"
        assert len(chain.steps) == 1

    def test_process_with_reflection(self):
        """Test processing with reflection enabled."""
        chain = ChainOfThought(id="chain_1")
        process = ThoughtProcess(chain=chain, enable_reflection=True)

        process.process("First thought")
        result = process.process("Second thought")

        assert "reflection" in result

    def test_finalize(self):
        """Test finalizing thought process."""
        chain = ChainOfThought(id="chain_1")
        process = ThoughtProcess(chain=chain)

        process.process("Thought 1")
        process.process("Thought 2")

        result = process.finalize()
        assert result["total_steps"] == 2
        assert result["ready_for_execution"] is True


class TestTaskDependency:
    """Tests for TaskDependency."""

    def test_dependency_creation(self):
        """Test creating task dependency."""
        dep = TaskDependency(
            task_id="task_2",
            depends_on=["task_1"],
            blocks=["task_3"]
        )
        assert dep.task_id == "task_2"
        assert "task_1" in dep.depends_on

    def test_check_satisfaction(self):
        """Test checking dependency satisfaction."""
        dep = TaskDependency(
            task_id="task_2",
            depends_on=["task_1"]
        )

        # Not satisfied yet
        assert dep.check_satisfaction(set()) is False

        # Now satisfied
        assert dep.check_satisfaction({"task_1"}) is True


class TestPersistentTaskMemory:
    """Tests for PersistentTaskMemory."""

    def test_memory_creation(self):
        """Test creating persistent memory."""
        memory = PersistentTaskMemory()
        assert memory.profile == "default"
        assert len(memory.tasks) == 0

    def test_add_task_to_memory(self):
        """Test adding task to memory."""
        memory = PersistentTaskMemory()
        task = PRDTask(id="task_1", title="Test", description="Desc")

        memory.add_task(task)

        assert "task_1" in memory.tasks
        assert "task_1" in memory.dependencies

    def test_add_chain_to_memory(self):
        """Test adding chain to memory."""
        memory = PersistentTaskMemory()
        chain = ChainOfThought(id="chain_1")

        memory.add_chain(chain)

        assert "chain_1" in memory.chains

    def test_get_status(self):
        """Test getting memory status."""
        memory = PersistentTaskMemory()
        task1 = PRDTask(id="task_1", title="Test 1", description="Desc", status="pending")
        task2 = PRDTask(id="task_2", title="Test 2", description="Desc", status="completed")

        memory.add_task(task1)
        memory.add_task(task2)

        status = memory.get_status()
        assert status["task_count"] == 2
        assert status["pending_tasks"] == 1
        assert status["completed_tasks"] == 1

    def test_save_and_load(self):
        """Test saving and loading memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = PersistentTaskMemory(data_dir=Path(tmpdir))
            task = PRDTask(id="task_1", title="Test", description="Desc")
            memory.add_task(task)

            # Save
            assert memory.save() is True

            # Load in new instance
            memory2 = PersistentTaskMemory(data_dir=Path(tmpdir))
            assert memory2.load() is True
            assert "task_1" in memory2.tasks


class TestStructuredWorkflow:
    """Tests for StructuredWorkflow."""

    def test_workflow_creation(self):
        """Test creating structured workflow."""
        workflow = StructuredWorkflow()
        assert workflow.mode == TaskWorkflowMode.PLANNING

    def test_plan_task(self):
        """Test entering planning mode."""
        workflow = StructuredWorkflow()
        chain = workflow.plan_task("Implement feature X")

        assert workflow.mode == TaskWorkflowMode.PLANNING
        assert chain is not None
        assert len(chain.steps) == 1

    def test_execute_task(self):
        """Test entering execution mode."""
        workflow = StructuredWorkflow()
        task = PRDTask(id="task_1", title="Test", description="Desc")

        result = workflow.execute_task(task)

        assert workflow.mode == TaskWorkflowMode.EXECUTION
        assert result["task_id"] == "task_1"
        assert task.status == "in_progress"

    def test_enter_research_mode(self):
        """Test entering research mode."""
        workflow = StructuredWorkflow()
        result = workflow.enter_research_mode("Authentication patterns")

        assert workflow.mode == TaskWorkflowMode.RESEARCH
        assert result["topic"] == "Authentication patterns"

    def test_reflect_on_task(self):
        """Test entering reflection mode."""
        workflow = StructuredWorkflow()
        task = PRDTask(id="task_1", title="Test", description="Desc")
        workflow._get_memory().add_task(task)

        result = workflow.reflect_on_task("task_1")

        assert workflow.mode == TaskWorkflowMode.REFLECTION
        assert result["task_id"] == "task_1"

    def test_complete_task(self):
        """Test completing a task."""
        workflow = StructuredWorkflow()
        task = PRDTask(id="task_1", title="Test", description="Desc")
        workflow._get_memory().add_task(task)

        result = workflow.complete_task("task_1")

        assert result is True
        assert task.status == "completed"

    def test_init_project_rules(self):
        """Test initializing project rules."""
        workflow = StructuredWorkflow()
        rules = {"max_file_size": 1000, "require_tests": True}

        workflow.init_project_rules(rules)

        assert workflow.project_rules == rules


# =============================================================================
# Semgrep Security Pattern Tests
# =============================================================================


class TestSecuritySeverity:
    """Tests for SecuritySeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert SecuritySeverity.INFO.value == "info"
        assert SecuritySeverity.WARNING.value == "warning"
        assert SecuritySeverity.ERROR.value == "error"
        assert SecuritySeverity.CRITICAL.value == "critical"


class TestRuleCategory:
    """Tests for RuleCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert RuleCategory.INJECTION.value == "injection"
        assert RuleCategory.AUTH.value == "authentication"
        assert RuleCategory.CRYPTO.value == "cryptography"
        assert RuleCategory.SECRETS.value == "secrets"


class TestSecurityFinding:
    """Tests for SecurityFinding."""

    def test_finding_creation(self):
        """Test creating security finding."""
        finding = SecurityFinding(
            id="finding_1",
            rule_id="sql-injection",
            file_path="app.py",
            line_start=10,
            line_end=10,
            severity=SecuritySeverity.CRITICAL,
            category=RuleCategory.INJECTION,
            message="Potential SQL injection"
        )

        assert finding.id == "finding_1"
        assert finding.severity == SecuritySeverity.CRITICAL
        assert finding.category == RuleCategory.INJECTION

    def test_finding_to_dict(self):
        """Test finding serialization."""
        finding = SecurityFinding(
            id="finding_1",
            rule_id="test-rule",
            file_path="test.py",
            line_start=1,
            line_end=5,
            severity=SecuritySeverity.WARNING,
            category=RuleCategory.BEST_PRACTICES,
            message="Test message"
        )

        data = finding.to_dict()
        assert data["severity"] == "warning"
        assert data["category"] == "best-practices"


class TestASTNode:
    """Tests for ASTNode."""

    def test_node_creation(self):
        """Test creating AST node."""
        node = ASTNode(
            node_type="function_definition",
            value="my_function",
            line=10
        )

        assert node.node_type == "function_definition"
        assert node.value == "my_function"

    def test_node_with_children(self):
        """Test node with children."""
        child1 = ASTNode(node_type="argument", value="arg1")
        child2 = ASTNode(node_type="argument", value="arg2")

        parent = ASTNode(
            node_type="function_definition",
            value="my_func",
            children=[child1, child2]
        )

        assert len(parent.children) == 2

    def test_find_nodes(self):
        """Test finding nodes by type."""
        child1 = ASTNode(node_type="argument", value="arg1")
        child2 = ASTNode(node_type="return_statement", value="x")

        func = ASTNode(
            node_type="function_definition",
            value="test",
            children=[child1, child2]
        )

        args = func.find_nodes("argument")
        assert len(args) == 1
        assert args[0].value == "arg1"


class TestSemgrepRule:
    """Tests for SemgrepRule."""

    def test_rule_creation(self):
        """Test creating Semgrep rule."""
        rule = SemgrepRule(
            id="hardcoded-password",
            pattern=r"password\s*=\s*['\"].*['\"]",
            message="Hardcoded password detected",
            severity=SecuritySeverity.CRITICAL,
            category=RuleCategory.SECRETS
        )

        assert rule.id == "hardcoded-password"
        assert rule.severity == SecuritySeverity.CRITICAL

    def test_rule_to_yaml(self):
        """Test rule YAML generation."""
        rule = SemgrepRule(
            id="test-rule",
            pattern=r"eval\(",
            message="Use of eval",
            severity=SecuritySeverity.WARNING
        )

        yaml = rule.to_yaml()
        assert "id: test-rule" in yaml
        assert "severity: WARNING" in yaml

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = SemgrepRule(
            id="test-rule",
            pattern=r"test",
            message="Test",
            languages=["python", "javascript"]
        )

        data = rule.to_dict()
        assert data["languages"] == ["python", "javascript"]


class TestASTAnalyzer:
    """Tests for ASTAnalyzer."""

    def test_analyzer_creation(self):
        """Test creating AST analyzer."""
        analyzer = ASTAnalyzer()
        assert len(analyzer._patterns) == 0

    def test_register_rule(self):
        """Test registering a rule."""
        analyzer = ASTAnalyzer()
        rule = SemgrepRule(
            id="test-rule",
            pattern=r"test",
            message="Test",
            languages=["python"]
        )

        analyzer.register_rule(rule)
        assert "python" in analyzer._patterns
        assert len(analyzer._patterns["python"]) == 1

    def test_parse_code(self):
        """Test parsing code to AST."""
        analyzer = ASTAnalyzer()
        code = """
def hello():
    pass

class MyClass:
    pass
"""

        ast = analyzer.parse_code(code)
        assert ast.node_type == "module"
        assert len(ast.children) == 2  # 1 function + 1 class

    def test_analyze_code(self):
        """Test analyzing code for security issues."""
        analyzer = ASTAnalyzer()
        rule = SemgrepRule(
            id="eval-usage",
            pattern=r"\beval\(",
            message="Dangerous eval usage",
            severity=SecuritySeverity.CRITICAL,
            category=RuleCategory.INJECTION
        )

        code = """
result = eval(user_input)
"""

        findings = analyzer.analyze(code, "python", [rule])
        assert len(findings) == 1
        assert findings[0].rule_id == "eval-usage"

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        analyzer = ASTAnalyzer()
        languages = analyzer.get_supported_languages()

        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages


class TestSecurityScanner:
    """Tests for SecurityScanner."""

    def test_scanner_creation(self):
        """Test creating security scanner."""
        scanner = SecurityScanner()
        # Default rules should be registered
        assert len(scanner.rules) > 0

    def test_scan_code_with_hardcoded_password(self):
        """Test scanning code with hardcoded password."""
        scanner = SecurityScanner()
        code = """
password = "secret123"
"""

        findings = scanner.scan_code(code)
        critical_findings = [f for f in findings if f.severity == SecuritySeverity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_scan_code_with_eval(self):
        """Test scanning code with eval usage."""
        scanner = SecurityScanner()
        code = """
result = eval(user_input)
"""

        findings = scanner.scan_code(code)
        assert any(f.rule_id == "eval-usage" for f in findings)

    def test_add_custom_rule(self):
        """Test adding custom rule to scanner."""
        scanner = SecurityScanner()
        initial_count = len(scanner.rules)

        custom_rule = SemgrepRule(
            id="custom-rule",
            pattern=r"print\(",
            message="Don't use print in production",
            severity=SecuritySeverity.WARNING,
            category=RuleCategory.BEST_PRACTICES
        )

        scanner.add_rule(custom_rule)
        assert len(scanner.rules) == initial_count + 1

    def test_filter_by_severity(self):
        """Test filtering findings by severity."""
        scanner = SecurityScanner()

        findings = [
            SecurityFinding(
                id="f1", rule_id="r1", file_path="a.py",
                line_start=1, line_end=1,
                severity=SecuritySeverity.INFO,
                category=RuleCategory.BEST_PRACTICES,
                message="Info"
            ),
            SecurityFinding(
                id="f2", rule_id="r2", file_path="a.py",
                line_start=2, line_end=2,
                severity=SecuritySeverity.CRITICAL,
                category=RuleCategory.INJECTION,
                message="Critical"
            )
        ]

        filtered = scanner.filter_by_severity(findings, SecuritySeverity.ERROR)
        assert len(filtered) == 1
        assert filtered[0].severity == SecuritySeverity.CRITICAL


class TestSecurityAudit:
    """Tests for SecurityAudit."""

    def test_audit_creation(self):
        """Test creating security audit."""
        audit = SecurityAudit(audit_id="audit_1")
        assert audit.audit_id == "audit_1"
        assert audit.passed is True
        assert len(audit.findings) == 0

    def test_add_findings(self):
        """Test adding findings to audit."""
        audit = SecurityAudit(audit_id="audit_1")

        finding = SecurityFinding(
            id="f1", rule_id="r1", file_path="a.py",
            line_start=1, line_end=1,
            severity=SecuritySeverity.CRITICAL,
            category=RuleCategory.INJECTION,
            message="Critical issue"
        )

        audit.add_findings([finding])

        assert len(audit.findings) == 1
        assert audit.passed is False  # Critical finding causes failure

    def test_complete_audit(self):
        """Test completing audit."""
        audit = SecurityAudit(audit_id="audit_1")
        audit.complete()

        assert audit.completed_at is not None

    def test_get_summary(self):
        """Test getting audit summary."""
        audit = SecurityAudit(audit_id="audit_1")
        audit.files_scanned = ["a.py", "b.py"]

        finding1 = SecurityFinding(
            id="f1", rule_id="r1", file_path="a.py",
            line_start=1, line_end=1,
            severity=SecuritySeverity.WARNING,
            category=RuleCategory.BEST_PRACTICES,
            message="Warning"
        )
        finding2 = SecurityFinding(
            id="f2", rule_id="r2", file_path="b.py",
            line_start=1, line_end=1,
            severity=SecuritySeverity.ERROR,
            category=RuleCategory.INJECTION,
            message="Error"
        )

        audit.add_findings([finding1, finding2])
        audit.complete()

        summary = audit.get_summary()
        assert summary["files_scanned"] == 2
        assert summary["total_findings"] == 2
        assert summary["severity_breakdown"]["warning"] == 1
        assert summary["severity_breakdown"]["error"] == 1

    def test_audit_to_dict(self):
        """Test audit serialization."""
        audit = SecurityAudit(audit_id="audit_1")
        audit.complete()

        data = audit.to_dict()
        assert data["audit_id"] == "audit_1"
        assert "summary" in data


# =============================================================================
# Integration Tests
# =============================================================================


class TestV109Integration:
    """Integration tests for V10.9 patterns."""

    def test_prd_to_execution_workflow(self):
        """Test complete PRD to execution workflow."""
        # 1. Create PRD workflow
        workflow = PRDWorkflow()

        # 2. Parse PRD content
        prd = """
        1. Design database schema
        2. Implement API endpoints
        3. Create frontend components
        """
        tasks = workflow.parse_prd(prd)
        assert len(tasks) == 3

        # 3. Get and execute first task
        task = workflow.get_next_task()
        assert task is not None

        # 4. Mark as complete
        workflow.set_task_status(task.id, "completed")
        assert task.status == "completed"

    def test_chain_of_thought_workflow(self):
        """Test chain of thought reasoning workflow."""
        # 1. Create structured workflow
        workflow = StructuredWorkflow()

        # 2. Plan task
        chain = workflow.plan_task("Implement authentication")

        # 3. Add reasoning steps
        process = ThoughtProcess(chain=chain)
        process.process("Analyzing authentication options")
        process.process("OAuth2 seems suitable", context={"option": "oauth2"})
        process.process("Implementation plan ready")

        # 4. Finalize
        result = process.finalize()
        assert result["total_steps"] == 4  # 1 from plan_task + 3 from process
        assert result["ready_for_execution"] is True

    def test_security_audit_workflow(self):
        """Test complete security audit workflow."""
        # 1. Create scanner
        scanner = SecurityScanner()

        # 2. Create audit
        audit = SecurityAudit(audit_id="audit_2026")

        # 3. Scan code
        vulnerable_code = """
password = "secret123"
result = eval(input())
DEBUG = True
"""
        findings = scanner.scan_code(vulnerable_code)
        audit.add_findings(findings)
        audit.files_scanned.append("app.py")

        # 4. Complete audit
        audit.complete()

        # 5. Check results
        summary = audit.get_summary()
        assert summary["total_findings"] >= 3
        assert audit.passed is False  # Has critical findings

    def test_task_with_security_audit(self):
        """Test combining task management with security audit."""
        # 1. Create task for security audit
        task = PRDTask(
            id="security_audit",
            title="Run security audit on codebase",
            description="Scan all Python files for vulnerabilities",
            acceptance_criteria=["No critical findings", "No SQL injection"]
        )

        # 2. Create workflow
        workflow = StructuredWorkflow()
        workflow._get_memory().add_task(task)

        # 3. Execute security audit
        scanner = SecurityScanner()
        audit = SecurityAudit(audit_id="task_audit")

        safe_code = """
def hello():
    return "Hello, World!"
"""
        findings = scanner.scan_code(safe_code)
        audit.add_findings(findings)
        audit.complete()

        # 4. Verify task based on audit
        if audit.passed:
            workflow.complete_task("security_audit")
            assert task.status == "completed"
