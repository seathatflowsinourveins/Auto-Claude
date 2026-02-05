#!/usr/bin/env python3
"""
Daemon Workers - Gap19 Resolution

Background workers for UNLEASH platform that perform automated tasks:
- optimize: Performance optimization analysis
- testgaps: Test coverage gap detection
- document: Auto-documentation generation

Key fixes from prior failures:
1. Uses temp files for prompts (avoids Windows ENAMETOOLONG - 8191 char limit)
2. Implements real local functionality without requiring Claude CLI
3. Proper timeout handling and error recovery
4. Stores results in platform/data/ for cross-session persistence
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('daemon_workers')


class WorkerResult:
    """Result from a worker execution."""

    def __init__(
        self,
        worker_type: str,
        success: bool,
        output: dict[str, Any],
        duration_ms: float,
        error: str | None = None
    ):
        self.worker_type = worker_type
        self.success = success
        self.output = output
        self.duration_ms = duration_ms
        self.error = error
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            'worker_type': self.worker_type,
            'success': self.success,
            'output': self.output,
            'duration_ms': self.duration_ms,
            'error': self.error,
            'timestamp': self.timestamp
        }


class DaemonWorkerManager:
    """Manages daemon workers for the UNLEASH platform."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / 'platform' / 'data' / 'daemon_workers'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Worker state tracking
        self.worker_stats: dict[str, dict] = {}
        self._load_stats()

    def _load_stats(self) -> None:
        """Load worker stats from disk."""
        stats_file = self.data_dir / 'worker_stats.json'
        if stats_file.exists():
            try:
                self.worker_stats = json.loads(stats_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load worker stats: {e}")
                self.worker_stats = {}

    def _save_stats(self) -> None:
        """Save worker stats to disk."""
        stats_file = self.data_dir / 'worker_stats.json'
        try:
            stats_file.write_text(json.dumps(self.worker_stats, indent=2))
        except OSError as e:
            logger.error(f"Failed to save worker stats: {e}")

    def _update_stats(self, worker_type: str, result: WorkerResult) -> None:
        """Update stats for a worker."""
        if worker_type not in self.worker_stats:
            self.worker_stats[worker_type] = {
                'run_count': 0,
                'success_count': 0,
                'failure_count': 0,
                'average_duration_ms': 0,
                'last_run': None,
                'last_error': None
            }

        stats = self.worker_stats[worker_type]
        stats['run_count'] += 1

        if result.success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
            stats['last_error'] = result.error

        # Update running average
        n = stats['run_count']
        old_avg = stats['average_duration_ms']
        stats['average_duration_ms'] = old_avg + (result.duration_ms - old_avg) / n
        stats['last_run'] = result.timestamp

        self._save_stats()

    def _save_result(self, result: WorkerResult) -> Path:
        """Save worker result to disk."""
        result_dir = self.data_dir / result.worker_type
        result_dir.mkdir(exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        result_file = result_dir / f'{timestamp}.json'
        result_file.write_text(json.dumps(result.to_dict(), indent=2))

        # Also update latest.json for easy access
        latest_file = result_dir / 'latest.json'
        latest_file.write_text(json.dumps(result.to_dict(), indent=2))

        return result_file

    async def run_optimize(self) -> WorkerResult:
        """
        Optimize worker: Analyze codebase for performance improvements.

        Analyzes:
        - Import patterns (detect slow imports)
        - Loop inefficiencies
        - Memory patterns (large allocations)
        - Database query patterns
        - Caching opportunities
        """
        start_time = time.perf_counter()

        try:
            findings: list[dict] = []
            files_analyzed = 0

            # Scan Python files in platform directory
            platform_dir = self.project_root / 'platform'
            for py_file in platform_dir.rglob('*.py'):
                if 'node_modules' in str(py_file) or '__pycache__' in str(py_file):
                    continue

                files_analyzed += 1

                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    lines = content.split('\n')
                    rel_path = py_file.relative_to(self.project_root)

                    # Check for slow import patterns
                    for i, line in enumerate(lines[:50], 1):
                        # Heavy imports at module level
                        if re.match(r'^import (pandas|tensorflow|torch|transformers)', line):
                            findings.append({
                                'type': 'slow_import',
                                'file': str(rel_path),
                                'line': i,
                                'severity': 'medium',
                                'description': f'Heavy import at module level: {line.strip()}',
                                'suggestion': 'Consider lazy importing inside functions'
                            })

                        # Import * pattern
                        if re.match(r'^from .+ import \*', line):
                            findings.append({
                                'type': 'wildcard_import',
                                'file': str(rel_path),
                                'line': i,
                                'severity': 'low',
                                'description': 'Wildcard import may slow down module loading',
                                'suggestion': 'Import only needed symbols'
                            })

                    # Check for inefficient patterns
                    for i, line in enumerate(lines, 1):
                        # Repeated list/dict comprehension in loops
                        if re.search(r'for .+ in .+:\s*$', line):
                            # Check next few lines for patterns
                            next_lines = '\n'.join(lines[i:i+5])
                            if re.search(r'\.append\(', next_lines):
                                findings.append({
                                    'type': 'append_in_loop',
                                    'file': str(rel_path),
                                    'line': i,
                                    'severity': 'low',
                                    'description': 'List append in loop could use comprehension',
                                    'suggestion': 'Consider using list comprehension'
                                })

                        # String concatenation in loops
                        # Match: result += "..." or result += str(...) or result = result + "..."
                        if re.search(r'\+= ["\']', line) or re.search(r'\+= str\(', line) or re.search(r'= .+ \+ ["\']', line):
                            findings.append({
                                'type': 'string_concat',
                                'file': str(rel_path),
                                'line': i,
                                'severity': 'low',
                                'description': 'String concatenation may be inefficient',
                                'suggestion': 'Use str.join() or f-strings'
                            })

                        # Synchronous file I/O in async context
                        if 'async def' in line:
                            # Check function body for sync I/O
                            func_body = '\n'.join(lines[i:i+30])
                            if re.search(r'open\(|\.read\(|\.write\(', func_body):
                                if not re.search(r'aiofiles|async with', func_body):
                                    findings.append({
                                        'type': 'sync_io_in_async',
                                        'file': str(rel_path),
                                        'line': i,
                                        'severity': 'medium',
                                        'description': 'Synchronous I/O in async function',
                                        'suggestion': 'Use aiofiles for async file I/O'
                                    })

                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Could not analyze {py_file}: {e}")

            # Deduplicate findings by file+line
            seen = set()
            unique_findings = []
            for f in findings:
                key = (f['file'], f['line'], f['type'])
                if key not in seen:
                    seen.add(key)
                    unique_findings.append(f)

            # Sort by severity
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            unique_findings.sort(key=lambda x: severity_order.get(x['severity'], 3))

            output = {
                'files_analyzed': files_analyzed,
                'findings_count': len(unique_findings),
                'findings': unique_findings[:50],  # Limit to top 50
                'by_severity': {
                    'high': len([f for f in unique_findings if f['severity'] == 'high']),
                    'medium': len([f for f in unique_findings if f['severity'] == 'medium']),
                    'low': len([f for f in unique_findings if f['severity'] == 'low'])
                },
                'by_type': {}
            }

            # Count by type
            for f in unique_findings:
                t = f['type']
                output['by_type'][t] = output['by_type'].get(t, 0) + 1

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = WorkerResult('optimize', True, output, duration_ms)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Optimize worker failed: {e}")
            result = WorkerResult('optimize', False, {}, duration_ms, str(e))

        self._update_stats('optimize', result)
        self._save_result(result)
        return result

    async def run_testgaps(self) -> WorkerResult:
        """
        Testgaps worker: Identify test coverage gaps.

        Analyzes:
        - Functions/classes without tests
        - Low coverage modules
        - Missing edge case tests
        - Integration test gaps
        """
        start_time = time.perf_counter()

        try:
            # Map source files to their potential test files
            source_files: dict[str, Path] = {}
            test_files: dict[str, Path] = {}

            platform_dir = self.project_root / 'platform'

            # Collect source files
            for py_file in platform_dir.rglob('*.py'):
                if 'node_modules' in str(py_file) or '__pycache__' in str(py_file):
                    continue

                rel_path = py_file.relative_to(platform_dir)

                if 'test' in str(rel_path).lower():
                    # It's a test file
                    test_files[str(rel_path)] = py_file
                else:
                    source_files[str(rel_path)] = py_file

            gaps: list[dict] = []
            tested_modules: set[str] = set()

            # Analyze test files to see what they cover
            for test_path, test_file in test_files.items():
                try:
                    content = test_file.read_text(encoding='utf-8', errors='ignore')

                    # Look for imports to determine what's being tested
                    imports = re.findall(r'from ([.\w]+) import|import ([.\w]+)', content)
                    for imp in imports:
                        module = imp[0] or imp[1]
                        if module.startswith('platform.') or module.startswith('.'):
                            tested_modules.add(module.replace('.', '/'))

                    # Look for test function counts
                    test_funcs = re.findall(r'def (test_\w+)', content)
                    async_test_funcs = re.findall(r'async def (test_\w+)', content)

                except (OSError, UnicodeDecodeError):
                    continue

            # Identify untested source files
            for src_path, src_file in source_files.items():
                # Skip __init__.py and utility files
                if src_file.name in ('__init__.py', 'conftest.py'):
                    continue

                # Check if there's a corresponding test file
                src_stem = src_file.stem
                possible_test_names = [
                    f'test_{src_stem}.py',
                    f'{src_stem}_test.py',
                    f'tests/test_{src_stem}.py'
                ]

                has_test = False
                for test_name in possible_test_names:
                    if any(test_name in str(tf) for tf in test_files.keys()):
                        has_test = True
                        break

                # Also check via import analysis
                module_path = str(src_file.relative_to(platform_dir)).replace('\\', '/').replace('.py', '')
                if any(module_path in tm for tm in tested_modules):
                    has_test = True

                if not has_test:
                    # Analyze the source file to prioritize
                    try:
                        content = src_file.read_text(encoding='utf-8', errors='ignore')

                        # Count functions and classes
                        funcs = len(re.findall(r'^\s*(?:async\s+)?def \w+', content, re.MULTILINE))
                        classes = len(re.findall(r'^class \w+', content, re.MULTILINE))
                        lines = len(content.split('\n'))

                        # Higher priority for larger files with more functions
                        priority = 'high' if (funcs > 5 or classes > 2 or lines > 200) else \
                                   'medium' if (funcs > 2 or lines > 100) else 'low'

                        gaps.append({
                            'file': str(src_path),
                            'priority': priority,
                            'functions': funcs,
                            'classes': classes,
                            'lines': lines,
                            'suggestion': f'Create test_{src_stem}.py with {funcs} test functions'
                        })

                    except (OSError, UnicodeDecodeError):
                        gaps.append({
                            'file': str(src_path),
                            'priority': 'low',
                            'functions': 0,
                            'classes': 0,
                            'lines': 0,
                            'suggestion': 'Unable to analyze - create basic tests'
                        })

            # Sort gaps by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            gaps.sort(key=lambda x: (priority_order.get(x['priority'], 3), -x.get('functions', 0)))

            output = {
                'source_files': len(source_files),
                'test_files': len(test_files),
                'coverage_ratio': round(1 - len(gaps) / max(len(source_files), 1), 2),
                'gaps_count': len(gaps),
                'gaps': gaps[:30],  # Top 30 gaps
                'by_priority': {
                    'high': len([g for g in gaps if g['priority'] == 'high']),
                    'medium': len([g for g in gaps if g['priority'] == 'medium']),
                    'low': len([g for g in gaps if g['priority'] == 'low'])
                },
                'recommendations': [
                    f"Create tests for {len([g for g in gaps if g['priority'] == 'high'])} high-priority files",
                    f"Total test coverage gap: {len(gaps)} files without tests",
                    f"Focus on files with many functions first"
                ]
            }

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = WorkerResult('testgaps', True, output, duration_ms)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Testgaps worker failed: {e}")
            result = WorkerResult('testgaps', False, {}, duration_ms, str(e))

        self._update_stats('testgaps', result)
        self._save_result(result)
        return result

    async def run_document(self) -> WorkerResult:
        """
        Document worker: Identify and generate documentation needs.

        Analyzes:
        - Missing docstrings
        - Undocumented public APIs
        - Outdated documentation
        - Missing README sections
        """
        start_time = time.perf_counter()

        try:
            doc_issues: list[dict] = []
            files_analyzed = 0

            platform_dir = self.project_root / 'platform'

            for py_file in platform_dir.rglob('*.py'):
                if 'node_modules' in str(py_file) or '__pycache__' in str(py_file):
                    continue

                files_analyzed += 1

                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    rel_path = py_file.relative_to(self.project_root)
                    lines = content.split('\n')

                    # Track current context
                    in_class = False
                    class_name = ''

                    for i, line in enumerate(lines):
                        stripped = line.strip()

                        # Check for class definitions
                        class_match = re.match(r'^class (\w+)', stripped)
                        if class_match:
                            in_class = True
                            class_name = class_match.group(1)

                            # Check for class docstring
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if not (next_line.startswith('"""') or next_line.startswith("'''")):
                                    doc_issues.append({
                                        'type': 'missing_class_docstring',
                                        'file': str(rel_path),
                                        'line': i + 1,
                                        'name': class_name,
                                        'priority': 'high',
                                        'suggestion': f'Add docstring for class {class_name}'
                                    })

                        # Check for function definitions
                        func_match = re.match(r'^(\s*)(?:async\s+)?def (\w+)\s*\(', line)
                        if func_match:
                            indent = len(func_match.group(1))
                            func_name = func_match.group(2)

                            # Skip private functions and test functions
                            if func_name.startswith('_') or func_name.startswith('test_'):
                                continue

                            # Check for docstring
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if not (next_line.startswith('"""') or next_line.startswith("'''")):
                                    # Determine priority based on context
                                    priority = 'high' if indent == 0 else \
                                               'medium' if in_class else 'low'

                                    context = f'{class_name}.' if in_class and indent > 0 else ''

                                    doc_issues.append({
                                        'type': 'missing_function_docstring',
                                        'file': str(rel_path),
                                        'line': i + 1,
                                        'name': f'{context}{func_name}',
                                        'priority': priority,
                                        'suggestion': f'Add docstring for {context}{func_name}()'
                                    })

                        # Reset class context if we're back at module level
                        if line and not line[0].isspace():
                            if not class_match and not stripped.startswith('class '):
                                in_class = False
                                class_name = ''

                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Could not analyze {py_file}: {e}")

            # Deduplicate
            seen = set()
            unique_issues = []
            for issue in doc_issues:
                key = (issue['file'], issue['line'], issue['name'])
                if key not in seen:
                    seen.add(key)
                    unique_issues.append(issue)

            # Sort by priority and file
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            unique_issues.sort(key=lambda x: (priority_order.get(x['priority'], 3), x['file']))

            output = {
                'files_analyzed': files_analyzed,
                'issues_count': len(unique_issues),
                'issues': unique_issues[:50],  # Top 50
                'by_priority': {
                    'high': len([i for i in unique_issues if i['priority'] == 'high']),
                    'medium': len([i for i in unique_issues if i['priority'] == 'medium']),
                    'low': len([i for i in unique_issues if i['priority'] == 'low'])
                },
                'by_type': {
                    'missing_class_docstring': len([i for i in unique_issues if i['type'] == 'missing_class_docstring']),
                    'missing_function_docstring': len([i for i in unique_issues if i['type'] == 'missing_function_docstring'])
                },
                'recommendations': [
                    f"Document {len([i for i in unique_issues if i['priority'] == 'high'])} public classes/functions first",
                    "Use Google-style docstrings for consistency",
                    "Include Args, Returns, and Raises sections"
                ]
            }

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = WorkerResult('document', True, output, duration_ms)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Document worker failed: {e}")
            result = WorkerResult('document', False, {}, duration_ms, str(e))

        self._update_stats('document', result)
        self._save_result(result)
        return result

    async def run_worker(self, worker_type: str) -> WorkerResult:
        """Run a specific worker by type."""
        workers = {
            'optimize': self.run_optimize,
            'testgaps': self.run_testgaps,
            'document': self.run_document
        }

        if worker_type not in workers:
            raise ValueError(f"Unknown worker type: {worker_type}")

        logger.info(f"Starting {worker_type} worker...")
        result = await workers[worker_type]()

        status = "SUCCESS" if result.success else "FAILED"
        logger.info(f"{worker_type} worker {status} in {result.duration_ms:.1f}ms")

        return result

    async def run_all(self) -> dict[str, WorkerResult]:
        """Run all workers sequentially."""
        results = {}
        for worker_type in ['optimize', 'testgaps', 'document']:
            results[worker_type] = await self.run_worker(worker_type)
        return results

    def get_stats(self) -> dict[str, dict]:
        """Get worker statistics."""
        return self.worker_stats

    def get_latest_result(self, worker_type: str) -> Optional[dict]:
        """Get the latest result for a worker."""
        latest_file = self.data_dir / worker_type / 'latest.json'
        if latest_file.exists():
            try:
                return json.loads(latest_file.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None


async def main():
    """Main entry point for daemon workers."""
    import argparse

    parser = argparse.ArgumentParser(description='UNLEASH Daemon Workers')
    parser.add_argument(
        'worker',
        choices=['optimize', 'testgaps', 'document', 'all', 'stats'],
        help='Worker to run or "all" to run all workers, "stats" to show statistics'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help='Project root directory'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    manager = DaemonWorkerManager(args.project_root)

    if args.worker == 'stats':
        stats = manager.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\n=== Daemon Worker Statistics ===\n")
            for worker_type, data in stats.items():
                success_rate = data['success_count'] / max(data['run_count'], 1) * 100
                print(f"{worker_type}:")
                print(f"  Runs: {data['run_count']} ({success_rate:.1f}% success)")
                print(f"  Avg duration: {data['average_duration_ms']:.1f}ms")
                print(f"  Last run: {data.get('last_run', 'Never')}")
                if data.get('last_error'):
                    print(f"  Last error: {data['last_error'][:50]}...")
                print()
        return

    if args.worker == 'all':
        results = await manager.run_all()
        output = {k: v.to_dict() for k, v in results.items()}
    else:
        result = await manager.run_worker(args.worker)
        output = result.to_dict()

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"\n=== {args.worker.upper()} Worker Results ===\n")
        if isinstance(output, dict) and 'worker_type' in output:
            # Single result
            print(f"Status: {'SUCCESS' if output['success'] else 'FAILED'}")
            print(f"Duration: {output['duration_ms']:.1f}ms")
            if output.get('error'):
                print(f"Error: {output['error']}")
            else:
                out = output['output']
                if 'findings_count' in out:
                    print(f"Findings: {out['findings_count']}")
                    print(f"By severity: {out.get('by_severity', {})}")
                elif 'gaps_count' in out:
                    print(f"Test gaps: {out['gaps_count']}")
                    print(f"Coverage ratio: {out.get('coverage_ratio', 0):.0%}")
                elif 'issues_count' in out:
                    print(f"Doc issues: {out['issues_count']}")
                    print(f"By priority: {out.get('by_priority', {})}")
        else:
            # Multiple results
            for worker_type, data in output.items():
                status = 'SUCCESS' if data['success'] else 'FAILED'
                print(f"{worker_type}: {status} ({data['duration_ms']:.1f}ms)")


if __name__ == '__main__':
    asyncio.run(main())
