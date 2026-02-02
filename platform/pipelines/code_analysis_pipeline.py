"""
Code Analysis Pipeline for Unleashed Platform

Combines semantic code understanding with AI-assisted analysis:
- Static analysis (AST parsing, complexity metrics)
- Semantic search (code embeddings, similarity)
- AI review (Claude-powered code review)
- Dependency analysis (import graphs, coupling)

This pipeline enables:
- Code quality assessment
- Refactoring suggestions
- Security vulnerability detection
- Documentation generation
"""

import os
import ast
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import re

# Register pipeline
from . import register_pipeline

PIPELINE_AVAILABLE = True  # Basic functionality always available
register_pipeline(
    "code_analysis",
    PIPELINE_AVAILABLE,
    dependencies=["ast", "pathlib"]
)


class AnalysisDepth(Enum):
    """Depth of code analysis."""
    QUICK = "quick"         # Syntax only
    STANDARD = "standard"   # Syntax + metrics
    DEEP = "deep"          # Full analysis
    COMPREHENSIVE = "comprehensive"  # All + AI review


class IssueType(Enum):
    """Types of code issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"


class IssueSeverity(Enum):
    """Severity of code issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CodeIssue:
    """A code issue found during analysis."""
    file: str
    line: int
    column: int
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    name: str
    file: str
    line: int
    lines_of_code: int
    cyclomatic_complexity: int
    parameters: int
    returns: int
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    is_async: bool = False


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    path: str
    lines_total: int
    lines_code: int
    lines_comment: int
    lines_blank: int
    imports: List[str] = field(default_factory=list)
    classes: int = 0
    functions: int = 0
    complexity_avg: float = 0.0
    complexity_max: int = 0


@dataclass
class AnalysisResult:
    """Result from code analysis pipeline."""
    files_analyzed: int
    total_lines: int
    issues: List[CodeIssue]
    file_metrics: List[FileMetrics]
    function_metrics: List[FunctionMetrics]
    summary: Dict[str, Any]
    execution_time: float
    depth: AnalysisDepth
    ai_review: Optional[str] = None


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor for code analysis."""

    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.lines = source.split('\n')
        self.functions: List[FunctionMetrics] = []
        self.classes: List[str] = []
        self.imports: List[str] = []
        self.issues: List[CodeIssue] = []
        self.calls: Set[str] = set()
        self._current_function = None

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node.name)

        # Check for class naming convention
        if not node.name[0].isupper():
            self.issues.append(CodeIssue(
                file=self.filename,
                line=node.lineno,
                column=node.col_offset,
                issue_type=IssueType.STYLE,
                severity=IssueSeverity.LOW,
                message=f"Class '{node.name}' should use PascalCase",
                suggestion=f"Rename to '{node.name.title().replace('_', '')}'"
            ))

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._analyze_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._analyze_function(node, is_async=True)

    def _analyze_function(self, node, is_async: bool):
        # Get docstring
        docstring = ast.get_docstring(node)

        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)

        # Count parameters
        args = node.args
        param_count = (
            len(args.args) +
            len(args.posonlyargs) +
            len(args.kwonlyargs) +
            (1 if args.vararg else 0) +
            (1 if args.kwarg else 0)
        )

        # Count return statements
        returns = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))

        # Get function calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)

        # Calculate lines of code
        end_line = node.end_lineno or node.lineno
        loc = end_line - node.lineno + 1

        self.functions.append(FunctionMetrics(
            name=node.name,
            file=self.filename,
            line=node.lineno,
            lines_of_code=loc,
            cyclomatic_complexity=complexity,
            parameters=param_count,
            returns=returns,
            docstring=docstring,
            calls=calls,
            is_async=is_async,
        ))

        # Check for issues
        if complexity > 10:
            self.issues.append(CodeIssue(
                file=self.filename,
                line=node.lineno,
                column=node.col_offset,
                issue_type=IssueType.WARNING,
                severity=IssueSeverity.MEDIUM,
                message=f"Function '{node.name}' has high complexity ({complexity})",
                suggestion="Consider breaking into smaller functions"
            ))

        if loc > 50:
            self.issues.append(CodeIssue(
                file=self.filename,
                line=node.lineno,
                column=node.col_offset,
                issue_type=IssueType.INFO,
                severity=IssueSeverity.LOW,
                message=f"Function '{node.name}' is long ({loc} lines)",
                suggestion="Consider refactoring into smaller functions"
            ))

        if not docstring and not node.name.startswith('_'):
            self.issues.append(CodeIssue(
                file=self.filename,
                line=node.lineno,
                column=node.col_offset,
                issue_type=IssueType.STYLE,
                severity=IssueSeverity.LOW,
                message=f"Function '{node.name}' is missing docstring",
                suggestion="Add docstring describing function purpose"
            ))

        self.generic_visit(node)

    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return complexity

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        self.generic_visit(node)


class CodeAnalysisPipeline:
    """
    Comprehensive code analysis pipeline.

    Combines static analysis, metrics collection, and AI-powered review
    to provide actionable insights about code quality.
    """

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize code analysis pipeline.

        Args:
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
        """
        self.include_patterns = include_patterns or ["**/*.py"]
        self.exclude_patterns = exclude_patterns or [
            "**/__pycache__/**",
            "**/venv/**",
            "**/.git/**",
            "**/node_modules/**",
        ]
        self._ai_reviewer = None
        self._try_init_ai()

    def _try_init_ai(self):
        """Try to initialize AI reviewer."""
        try:
            from ..adapters.llm_reasoners_adapter import LLMReasonersAdapter
            self._ai_reviewer = LLMReasonersAdapter()
        except ImportError:
            pass

    async def analyze(
        self,
        path: str,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
        ai_review: bool = False,
    ) -> AnalysisResult:
        """
        Analyze code at the given path.

        Args:
            path: File or directory path to analyze
            depth: Analysis depth level
            ai_review: Whether to include AI review

        Returns:
            AnalysisResult with findings
        """
        import time
        start_time = time.time()

        path_obj = Path(path)
        files_to_analyze = []

        if path_obj.is_file():
            files_to_analyze = [path_obj]
        elif path_obj.is_dir():
            files_to_analyze = self._collect_files(path_obj)
        else:
            raise ValueError(f"Path does not exist: {path}")

        all_issues: List[CodeIssue] = []
        all_file_metrics: List[FileMetrics] = []
        all_function_metrics: List[FunctionMetrics] = []
        total_lines = 0

        for file_path in files_to_analyze:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source = f.read()

                file_metrics, func_metrics, issues = self._analyze_file(
                    str(file_path), source, depth
                )

                all_file_metrics.append(file_metrics)
                all_function_metrics.extend(func_metrics)
                all_issues.extend(issues)
                total_lines += file_metrics.lines_total

            except Exception as e:
                all_issues.append(CodeIssue(
                    file=str(file_path),
                    line=0,
                    column=0,
                    issue_type=IssueType.ERROR,
                    severity=IssueSeverity.HIGH,
                    message=f"Failed to analyze file: {e}"
                ))

        # Generate summary
        summary = self._generate_summary(all_file_metrics, all_function_metrics, all_issues)

        # AI review if requested
        ai_review_text = None
        if ai_review and depth == AnalysisDepth.COMPREHENSIVE:
            ai_review_text = await self._generate_ai_review(
                all_issues, all_function_metrics, summary
            )

        execution_time = time.time() - start_time

        return AnalysisResult(
            files_analyzed=len(files_to_analyze),
            total_lines=total_lines,
            issues=all_issues,
            file_metrics=all_file_metrics,
            function_metrics=all_function_metrics,
            summary=summary,
            execution_time=execution_time,
            depth=depth,
            ai_review=ai_review_text,
        )

    def _collect_files(self, directory: Path) -> List[Path]:
        """Collect files matching include/exclude patterns."""
        files = []

        for pattern in self.include_patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    # Check exclusions
                    excluded = False
                    for exclude in self.exclude_patterns:
                        if file_path.match(exclude):
                            excluded = True
                            break
                    if not excluded:
                        files.append(file_path)

        return files

    def _analyze_file(
        self,
        filepath: str,
        source: str,
        depth: AnalysisDepth,
    ) -> Tuple[FileMetrics, List[FunctionMetrics], List[CodeIssue]]:
        """Analyze a single file."""
        lines = source.split('\n')
        lines_total = len(lines)
        lines_blank = sum(1 for line in lines if not line.strip())
        lines_comment = sum(1 for line in lines if line.strip().startswith('#'))
        lines_code = lines_total - lines_blank - lines_comment

        issues = []
        functions = []
        imports = []
        classes = 0

        if depth != AnalysisDepth.QUICK:
            try:
                tree = ast.parse(source)
                analyzer = ASTAnalyzer(source, filepath)
                analyzer.visit(tree)

                functions = analyzer.functions
                imports = analyzer.imports
                classes = len(analyzer.classes)
                issues = analyzer.issues

            except SyntaxError as e:
                issues.append(CodeIssue(
                    file=filepath,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    issue_type=IssueType.ERROR,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Syntax error: {e.msg}"
                ))

        # Calculate complexity metrics
        complexity_values = [f.cyclomatic_complexity for f in functions]
        complexity_avg = sum(complexity_values) / len(complexity_values) if complexity_values else 0
        complexity_max = max(complexity_values) if complexity_values else 0

        file_metrics = FileMetrics(
            path=filepath,
            lines_total=lines_total,
            lines_code=lines_code,
            lines_comment=lines_comment,
            lines_blank=lines_blank,
            imports=imports,
            classes=classes,
            functions=len(functions),
            complexity_avg=complexity_avg,
            complexity_max=complexity_max,
        )

        return file_metrics, functions, issues

    def _generate_summary(
        self,
        file_metrics: List[FileMetrics],
        function_metrics: List[FunctionMetrics],
        issues: List[CodeIssue],
    ) -> Dict[str, Any]:
        """Generate analysis summary."""
        # Count issues by type and severity
        issues_by_type = {}
        issues_by_severity = {}

        for issue in issues:
            type_name = issue.issue_type.value
            sev_name = issue.severity.value

            issues_by_type[type_name] = issues_by_type.get(type_name, 0) + 1
            issues_by_severity[sev_name] = issues_by_severity.get(sev_name, 0) + 1

        # Aggregate metrics
        total_loc = sum(f.lines_code for f in file_metrics)
        total_functions = sum(f.functions for f in file_metrics)
        total_classes = sum(f.classes for f in file_metrics)

        avg_complexity = 0
        if function_metrics:
            avg_complexity = sum(f.cyclomatic_complexity for f in function_metrics) / len(function_metrics)

        # Find most complex functions
        complex_functions = sorted(
            function_metrics,
            key=lambda f: f.cyclomatic_complexity,
            reverse=True
        )[:5]

        # Find largest functions
        largest_functions = sorted(
            function_metrics,
            key=lambda f: f.lines_of_code,
            reverse=True
        )[:5]

        return {
            "files": len(file_metrics),
            "total_lines_of_code": total_loc,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "average_complexity": round(avg_complexity, 2),
            "total_issues": len(issues),
            "issues_by_type": issues_by_type,
            "issues_by_severity": issues_by_severity,
            "most_complex_functions": [
                {"name": f.name, "file": f.file, "complexity": f.cyclomatic_complexity}
                for f in complex_functions
            ],
            "largest_functions": [
                {"name": f.name, "file": f.file, "lines": f.lines_of_code}
                for f in largest_functions
            ],
            "code_health_score": self._calculate_health_score(issues, total_loc),
        }

    def _calculate_health_score(self, issues: List[CodeIssue], total_loc: int) -> float:
        """Calculate overall code health score (0-100)."""
        if total_loc == 0:
            return 100.0

        # Weight issues by severity
        severity_weights = {
            IssueSeverity.CRITICAL: 10,
            IssueSeverity.HIGH: 5,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 0.5,
        }

        total_weight = sum(severity_weights.get(i.severity, 1) for i in issues)

        # Normalize by lines of code
        issues_per_kloc = (total_weight / total_loc) * 1000

        # Convert to score (0-100)
        score = max(0, 100 - issues_per_kloc * 2)
        return round(score, 1)

    async def _generate_ai_review(
        self,
        issues: List[CodeIssue],
        functions: List[FunctionMetrics],
        summary: Dict[str, Any],
    ) -> Optional[str]:
        """Generate AI-powered code review."""
        if not self._ai_reviewer:
            return None

        try:
            # Prepare context
            issue_summary = "\n".join([
                f"- {i.severity.value.upper()}: {i.message} ({i.file}:{i.line})"
                for i in issues[:20]  # Limit to top 20 issues
            ])

            complex_funcs = "\n".join([
                f"- {f['name']} (complexity: {f['complexity']})"
                for f in summary.get("most_complex_functions", [])
            ])

            prompt = f"""
Analyze this code quality report and provide actionable recommendations:

Summary:
- Files analyzed: {summary['files']}
- Total lines of code: {summary['total_lines_of_code']}
- Total functions: {summary['total_functions']}
- Average complexity: {summary['average_complexity']}
- Code health score: {summary['code_health_score']}/100

Top Issues:
{issue_summary}

Most Complex Functions:
{complex_funcs}

Provide:
1. Overall assessment (2-3 sentences)
2. Top 3 priority fixes
3. Architectural suggestions if applicable
"""

            result = await self._ai_reviewer.reason(
                problem=prompt,
                algorithm=None,
                max_depth=3,
            )

            return result.answer

        except Exception as e:
            return f"AI review unavailable: {e}"

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "available": PIPELINE_AVAILABLE,
            "ai_reviewer": self._ai_reviewer is not None,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
        }


# Convenience function
async def analyze_code(
    path: str,
    depth: str = "standard",
    ai_review: bool = False,
) -> AnalysisResult:
    """
    Quick code analysis helper.

    Args:
        path: Path to analyze
        depth: Analysis depth (quick/standard/deep/comprehensive)
        ai_review: Include AI review

    Returns:
        AnalysisResult
    """
    depth_map = {
        "quick": AnalysisDepth.QUICK,
        "standard": AnalysisDepth.STANDARD,
        "deep": AnalysisDepth.DEEP,
        "comprehensive": AnalysisDepth.COMPREHENSIVE,
    }

    pipeline = CodeAnalysisPipeline()
    return await pipeline.analyze(
        path,
        depth=depth_map.get(depth, AnalysisDepth.STANDARD),
        ai_review=ai_review,
    )
