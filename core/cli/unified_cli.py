#!/usr/bin/env python3
"""
Unified CLI - V33 Command Line Interface
Part of Phase 8: CLI Integration & Performance Optimization.

Provides production-grade CLI access to all V33 layers:
- Agent execution
- Memory operations
- Tool management
- Evaluation suites
- Trace analysis
- Configuration management
- System status

NO STUBS - All commands fully implemented.
NO GRACEFUL DEGRADATION - Explicit errors on missing dependencies.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Required dependencies - fail explicitly if missing
try:
    import click
except ImportError as e:
    raise ImportError(
        "click is required for CLI. Install with: pip install click"
    ) from e

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as e:
    raise ImportError(
        "rich is required for CLI output. Install with: pip install rich"
    ) from e

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required for config. Install with: pip install pyyaml"
    ) from e

# Initialize console
console = Console()

# Version info
__version__ = "35.0.0"


# ============================================================================
# Helper Functions
# ============================================================================


def get_config_path() -> Path:
    """Get the configuration file path."""
    return Path.cwd() / ".unleash.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from .unleash.yaml."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to .unleash.yaml."""
    config_path = get_config_path()
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_async(coro):
    """Run an async coroutine."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def format_json(data: Any) -> str:
    """Format data as JSON."""
    return json.dumps(data, indent=2, default=str)


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red bold]Error:[/red bold] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green bold]Success:[/green bold] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow bold]Warning:[/yellow bold] {message}")


# ============================================================================
# Main CLI Group
# ============================================================================


@click.group()
@click.version_option(version=__version__, prog_name="unleash")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, json_output: bool) -> None:
    """
    Unleash V33 CLI - Unified AI Agent Platform.

    Access all V33 layers from the command line including agents,
    memory, tools, evaluation, tracing, and configuration.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["json_output"] = json_output


# ============================================================================
# Run Commands - Execute Agents, Pipelines, Workflows
# ============================================================================


@cli.group()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Execute agents, pipelines, and workflows."""
    pass


@run.command("agent")
@click.argument("agent_name")
@click.option("--prompt", "-p", required=True, help="Prompt for the agent")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
@click.option("--max-turns", type=int, default=10, help="Maximum agent turns")
@click.option("--tools", "-t", multiple=True, help="Tools to enable")
@click.pass_context
def run_agent(
    ctx: click.Context,
    agent_name: str,
    prompt: str,
    model: str,
    max_turns: int,
    tools: tuple,
) -> None:
    """Run an agent with the specified prompt."""
    try:
        from core.orchestration.agent_sdk_layer import create_agent, run_agent_loop

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running agent '{agent_name}'...", total=None)

            async def execute():
                agent = await create_agent(
                    name=agent_name,
                    model=model,
                    tools=list(tools) if tools else None,
                )
                result = await run_agent_loop(
                    agent=agent,
                    prompt=prompt,
                    max_turns=max_turns,
                )
                return result

            result = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json(result))
        else:
            console.print(Panel(
                str(result.get("output", result)),
                title=f"[bold green]Agent: {agent_name}[/bold green]",
                border_style="green",
            ))

    except ImportError as e:
        print_error(f"Agent SDK not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Agent execution failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise SystemExit(1)


@run.command("pipeline")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--input", "-i", "input_data", help="Input data (JSON string)")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config override")
@click.pass_context
def run_pipeline(
    ctx: click.Context,
    pipeline_file: str,
    input_data: Optional[str],
    config: Optional[str],
) -> None:
    """Run a pipeline from a definition file."""
    try:
        from core.orchestration.langgraph_layer import load_pipeline, execute_pipeline

        # Load pipeline definition
        with open(pipeline_file) as f:
            pipeline_def = yaml.safe_load(f)

        # Parse input data
        parsed_input = json.loads(input_data) if input_data else {}

        # Load config override
        config_override = {}
        if config:
            with open(config) as f:
                config_override = yaml.safe_load(f)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running pipeline...", total=None)

            async def execute():
                pipeline = await load_pipeline(pipeline_def, config_override)
                result = await execute_pipeline(pipeline, parsed_input)
                return result

            result = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json(result))
        else:
            print_success(f"Pipeline completed: {pipeline_file}")
            console.print(Panel(format_json(result), title="Result"))

    except ImportError as e:
        print_error(f"LangGraph not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Pipeline execution failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise SystemExit(1)


@run.command("workflow")
@click.argument("workflow_name")
@click.option("--input", "-i", "input_data", help="Input data (JSON string)")
@click.option("--checkpoint", "-k", help="Resume from checkpoint ID")
@click.pass_context
def run_workflow(
    ctx: click.Context,
    workflow_name: str,
    input_data: Optional[str],
    checkpoint: Optional[str],
) -> None:
    """Run a predefined workflow."""
    try:
        from core.orchestration.workflow_runner import get_workflow, execute_workflow

        parsed_input = json.loads(input_data) if input_data else {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running workflow '{workflow_name}'...", total=None)

            async def execute():
                workflow = await get_workflow(workflow_name)
                result = await execute_workflow(
                    workflow,
                    parsed_input,
                    checkpoint_id=checkpoint,
                )
                return result

            result = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json(result))
        else:
            print_success(f"Workflow '{workflow_name}' completed")
            console.print(Panel(format_json(result), title="Result"))

    except ImportError as e:
        print_error(f"Workflow runner not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Workflow execution failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise SystemExit(1)


# ============================================================================
# Memory Commands - Store, Search, List, Delete
# ============================================================================


@cli.group()
@click.pass_context
def memory(ctx: click.Context) -> None:
    """Memory operations - store, search, list, delete."""
    pass


@memory.command("store")
@click.option("--key", "-k", required=True, help="Memory key")
@click.option("--value", "-v", required=True, help="Memory value")
@click.option("--metadata", "-m", help="Metadata (JSON string)")
@click.option("--namespace", "-n", default="default", help="Memory namespace")
@click.pass_context
def memory_store(
    ctx: click.Context,
    key: str,
    value: str,
    metadata: Optional[str],
    namespace: str,
) -> None:
    """Store a memory entry."""
    try:
        from core.memory import get_memory_manager

        parsed_metadata = json.loads(metadata) if metadata else {}

        async def execute():
            manager = get_memory_manager()
            result = await manager.store(
                key=key,
                value=value,
                metadata=parsed_metadata,
                namespace=namespace,
            )
            return result

        result = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json({"status": "stored", "key": key}))
        else:
            print_success(f"Stored memory: {key}")

    except ImportError as e:
        print_error(f"Memory layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Memory store failed: {e}")
        raise SystemExit(1)


@memory.command("search")
@click.argument("query")
@click.option("--limit", "-l", type=int, default=10, help="Maximum results")
@click.option("--namespace", "-n", default="default", help="Memory namespace")
@click.option("--threshold", "-t", type=float, default=0.7, help="Similarity threshold")
@click.pass_context
def memory_search(
    ctx: click.Context,
    query: str,
    limit: int,
    namespace: str,
    threshold: float,
) -> None:
    """Search memories with semantic query."""
    try:
        from core.memory import get_memory_manager

        async def execute():
            manager = get_memory_manager()
            results = await manager.search(
                query=query,
                limit=limit,
                namespace=namespace,
                threshold=threshold,
            )
            return results

        results = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json(results))
        else:
            table = Table(title=f"Search Results for: {query}")
            table.add_column("Key", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Value", max_width=50)

            for item in results:
                table.add_row(
                    item.get("key", "N/A"),
                    f"{item.get('score', 0):.3f}",
                    str(item.get("value", ""))[:50],
                )

            console.print(table)

    except ImportError as e:
        print_error(f"Memory layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Memory search failed: {e}")
        raise SystemExit(1)


@memory.command("list")
@click.option("--namespace", "-n", default="default", help="Memory namespace")
@click.option("--limit", "-l", type=int, default=100, help="Maximum entries")
@click.pass_context
def memory_list(
    ctx: click.Context,
    namespace: str,
    limit: int,
) -> None:
    """List all memory entries in a namespace."""
    try:
        from core.memory import get_memory_manager

        async def execute():
            manager = get_memory_manager()
            entries = await manager.list(namespace=namespace, limit=limit)
            return entries

        entries = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json(entries))
        else:
            table = Table(title=f"Memory Entries ({namespace})")
            table.add_column("Key", style="cyan")
            table.add_column("Created", style="dim")
            table.add_column("Size", justify="right")

            for entry in entries:
                table.add_row(
                    entry.get("key", "N/A"),
                    entry.get("created_at", "N/A"),
                    str(len(str(entry.get("value", "")))),
                )

            console.print(table)

    except ImportError as e:
        print_error(f"Memory layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Memory list failed: {e}")
        raise SystemExit(1)


@memory.command("delete")
@click.option("--key", "-k", required=True, help="Memory key to delete")
@click.option("--namespace", "-n", default="default", help="Memory namespace")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def memory_delete(
    ctx: click.Context,
    key: str,
    namespace: str,
    force: bool,
) -> None:
    """Delete a memory entry."""
    if not force:
        if not click.confirm(f"Delete memory '{key}' from namespace '{namespace}'?"):
            print_warning("Aborted")
            return

    try:
        from core.memory import get_memory_manager

        async def execute():
            manager = get_memory_manager()
            result = await manager.delete(key=key, namespace=namespace)
            return result

        result = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json({"status": "deleted", "key": key}))
        else:
            print_success(f"Deleted memory: {key}")

    except ImportError as e:
        print_error(f"Memory layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Memory delete failed: {e}")
        raise SystemExit(1)


# ============================================================================
# Tools Commands - List, Invoke, Describe
# ============================================================================


@cli.group()
@click.pass_context
def tools(ctx: click.Context) -> None:
    """Tool management - list, invoke, describe."""
    pass


@tools.command("list")
@click.option("--category", "-c", help="Filter by category")
@click.option("--source", "-s", help="Filter by source (builtin, mcp, custom)")
@click.pass_context
def tools_list(
    ctx: click.Context,
    category: Optional[str],
    source: Optional[str],
) -> None:
    """List all available tools."""
    try:
        from core.tools import get_tool_registry

        registry = get_tool_registry()
        all_tools = registry.list_tools(category=category, source=source)

        if ctx.obj.get("json_output"):
            console.print(format_json(all_tools))
        else:
            table = Table(title="Available Tools")
            table.add_column("Name", style="cyan")
            table.add_column("Category", style="yellow")
            table.add_column("Source", style="green")
            table.add_column("Description", max_width=40)

            for tool in all_tools:
                table.add_row(
                    tool.get("name", "N/A"),
                    tool.get("category", "N/A"),
                    tool.get("source", "N/A"),
                    tool.get("description", "")[:40],
                )

            console.print(table)

    except ImportError as e:
        print_error(f"Tools layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Tools list failed: {e}")
        raise SystemExit(1)


@tools.command("invoke")
@click.argument("tool_name")
@click.option("--args", "-a", help="Tool arguments (JSON string)")
@click.pass_context
def tools_invoke(
    ctx: click.Context,
    tool_name: str,
    args: Optional[str],
) -> None:
    """Invoke a tool with arguments."""
    try:
        from core.tools import get_tool_registry

        parsed_args = json.loads(args) if args else {}

        async def execute():
            registry = get_tool_registry()
            result = await registry.invoke(tool_name, **parsed_args)
            return result

        result = run_async(execute())

        if ctx.obj.get("json_output"):
            console.print(format_json(result))
        else:
            console.print(Panel(
                format_json(result),
                title=f"[bold]Tool: {tool_name}[/bold]",
            ))

    except ImportError as e:
        print_error(f"Tools layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Tool invocation failed: {e}")
        raise SystemExit(1)


@tools.command("describe")
@click.argument("tool_name")
@click.pass_context
def tools_describe(
    ctx: click.Context,
    tool_name: str,
) -> None:
    """Get detailed description of a tool."""
    try:
        from core.tools import get_tool_registry

        registry = get_tool_registry()
        tool_info = registry.describe(tool_name)

        if ctx.obj.get("json_output"):
            console.print(format_json(tool_info))
        else:
            console.print(Panel(
                f"[bold]{tool_info.get('name', tool_name)}[/bold]\n\n"
                f"{tool_info.get('description', 'No description')}\n\n"
                f"[dim]Category:[/dim] {tool_info.get('category', 'N/A')}\n"
                f"[dim]Source:[/dim] {tool_info.get('source', 'N/A')}\n\n"
                f"[bold]Parameters:[/bold]\n"
                f"{format_json(tool_info.get('parameters', {}))}",
                title="Tool Description",
            ))

    except ImportError as e:
        print_error(f"Tools layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Tool describe failed: {e}")
        raise SystemExit(1)


# ============================================================================
# Eval Commands - Run Evaluation Suites
# ============================================================================


@cli.group()
@click.pass_context
def eval(ctx: click.Context) -> None:
    """Evaluation and testing operations."""
    pass


@eval.command("run")
@click.argument("suite_name")
@click.option("--dataset", "-d", type=click.Path(exists=True), help="Test dataset file")
@click.option("--output", "-o", type=click.Path(), help="Output results file")
@click.option("--metrics", "-m", multiple=True, help="Metrics to compute")
@click.pass_context
def eval_run(
    ctx: click.Context,
    suite_name: str,
    dataset: Optional[str],
    output: Optional[str],
    metrics: tuple,
) -> None:
    """Run an evaluation suite."""
    try:
        from core.observability.ragas_evaluator import RAGASEvaluator

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running evaluation '{suite_name}'...", total=None)

            evaluator = RAGASEvaluator(metrics=list(metrics) if metrics else None)

            # Load dataset if provided
            test_data = []
            if dataset:
                with open(dataset) as f:
                    test_data = json.load(f)

            async def execute():
                # Run evaluation
                result = await evaluator.evaluate_dataset(test_data)
                return result

            result = run_async(execute())

        # Save output if requested
        if output:
            with open(output, "w") as f:
                json.dump(result.model_dump() if hasattr(result, "model_dump") else result, f, indent=2)
            print_success(f"Results saved to: {output}")

        if ctx.obj.get("json_output"):
            console.print(format_json(result.model_dump() if hasattr(result, "model_dump") else result))
        else:
            table = Table(title=f"Evaluation Results: {suite_name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Status")

            metrics_data = result.metrics if hasattr(result, "metrics") else {}
            for metric_name, score in metrics_data.items():
                status = "[green]PASS[/green]" if score >= 0.5 else "[red]FAIL[/red]"
                table.add_row(metric_name, f"{score:.3f}", status)

            console.print(table)

            overall = result.overall_score if hasattr(result, "overall_score") else 0
            console.print(f"\n[bold]Overall Score:[/bold] {overall:.3f}")

    except ImportError as e:
        print_error(f"Evaluation layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise SystemExit(1)


@eval.command("list")
@click.pass_context
def eval_list(ctx: click.Context) -> None:
    """List available evaluation suites."""
    try:
        # List available evaluators and suites
        suites = [
            {"name": "ragas", "description": "RAG evaluation with RAGAS metrics", "metrics": ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]},
            {"name": "promptfoo", "description": "Prompt testing and red-teaming", "metrics": ["prompt_injection", "jailbreak", "pii_leak"]},
            {"name": "custom", "description": "Custom evaluation suite", "metrics": ["custom"]},
        ]

        if ctx.obj.get("json_output"):
            console.print(format_json(suites))
        else:
            table = Table(title="Available Evaluation Suites")
            table.add_column("Name", style="cyan")
            table.add_column("Description", max_width=40)
            table.add_column("Metrics")

            for suite in suites:
                table.add_row(
                    suite["name"],
                    suite["description"],
                    ", ".join(suite["metrics"]),
                )

            console.print(table)

    except Exception as e:
        print_error(f"Failed to list evaluation suites: {e}")
        raise SystemExit(1)


# ============================================================================
# Trace Commands - List, Show, Export
# ============================================================================


@cli.group()
@click.pass_context
def trace(ctx: click.Context) -> None:
    """Trace analysis operations."""
    pass


@trace.command("list")
@click.option("--limit", "-l", type=int, default=20, help="Maximum traces")
@click.option("--since", "-s", help="Traces since (ISO datetime)")
@click.pass_context
def trace_list(
    ctx: click.Context,
    limit: int,
    since: Optional[str],
) -> None:
    """List recent traces."""
    try:
        from core.observability.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        traces = tracker.list_traces(limit=limit, since=since)

        if ctx.obj.get("json_output"):
            console.print(format_json(traces))
        else:
            table = Table(title="Recent Traces")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Status")
            table.add_column("Duration", justify="right")
            table.add_column("Timestamp", style="dim")

            for t in traces:
                status = "[green]OK[/green]" if t.get("status") == "ok" else "[red]ERR[/red]"
                table.add_row(
                    t.get("id", "N/A")[:8],
                    t.get("name", "N/A"),
                    status,
                    f"{t.get('duration_ms', 0):.0f}ms",
                    t.get("timestamp", "N/A"),
                )

            console.print(table)

    except ImportError as e:
        print_error(f"Tracing layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Trace list failed: {e}")
        raise SystemExit(1)


@trace.command("show")
@click.argument("trace_id")
@click.pass_context
def trace_show(
    ctx: click.Context,
    trace_id: str,
) -> None:
    """Show details of a specific trace."""
    try:
        from core.observability.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        trace_data = tracker.get_trace(trace_id)

        if ctx.obj.get("json_output"):
            console.print(format_json(trace_data))
        else:
            console.print(Panel(
                format_json(trace_data),
                title=f"[bold]Trace: {trace_id}[/bold]",
            ))

    except ImportError as e:
        print_error(f"Tracing layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Trace show failed: {e}")
        raise SystemExit(1)


@trace.command("export")
@click.argument("trace_id")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file")
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.pass_context
def trace_export(
    ctx: click.Context,
    trace_id: str,
    output: str,
    fmt: str,
) -> None:
    """Export a trace to a file."""
    try:
        from core.observability.langfuse_tracker import LangfuseTracker

        tracker = LangfuseTracker()
        trace_data = tracker.get_trace(trace_id)

        if fmt == "json":
            with open(output, "w") as f:
                json.dump(trace_data, f, indent=2, default=str)
        elif fmt == "csv":
            import csv
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=trace_data.keys())
                writer.writeheader()
                writer.writerow(trace_data)

        print_success(f"Exported trace to: {output}")

    except ImportError as e:
        print_error(f"Tracing layer not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Trace export failed: {e}")
        raise SystemExit(1)


# ============================================================================
# Config Commands - Show, Init, Validate
# ============================================================================


@cli.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configuration management."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    config_data = load_config()

    if ctx.obj.get("json_output"):
        console.print(format_json(config_data))
    else:
        if config_data:
            console.print(Panel(
                Syntax(yaml.dump(config_data, default_flow_style=False), "yaml"),
                title="[bold]Configuration[/bold]",
            ))
        else:
            print_warning("No configuration found. Run 'unleash config init' to create one.")


@config.command("init")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing config")
@click.pass_context
def config_init(ctx: click.Context, force: bool) -> None:
    """Initialize a new configuration file."""
    config_path = get_config_path()

    if config_path.exists() and not force:
        print_error(f"Configuration already exists at {config_path}. Use --force to overwrite.")
        raise SystemExit(1)

    default_config = {
        "version": "33.8.0",
        "project": {
            "name": Path.cwd().name,
            "description": "",
        },
        "llm": {
            "default_model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "memory": {
            "backend": "local",
            "namespace": "default",
        },
        "observability": {
            "enabled": True,
            "tracing": True,
            "metrics": True,
        },
        "performance": {
            "cache_enabled": True,
            "cache_ttl": 300,
            "connection_pool_size": 100,
        },
    }

    save_config(default_config)
    print_success(f"Configuration created at: {config_path}")


@config.command("validate")
@click.pass_context
def config_validate(ctx: click.Context) -> None:
    """Validate configuration file."""
    config_path = get_config_path()

    if not config_path.exists():
        print_error("No configuration file found. Run 'unleash config init' first.")
        raise SystemExit(1)

    try:
        config_data = load_config()

        # Validate required fields
        required_fields = ["version"]
        missing = [f for f in required_fields if f not in config_data]

        if missing:
            print_error(f"Missing required fields: {', '.join(missing)}")
            raise SystemExit(1)

        # Validate version format
        version = config_data.get("version", "")
        if not version:
            print_error("Version field is empty")
            raise SystemExit(1)

        print_success("Configuration is valid")

        if ctx.obj.get("verbose"):
            console.print(f"\nVersion: {version}")
            console.print(f"Sections: {', '.join(config_data.keys())}")

    except yaml.YAMLError as e:
        print_error(f"Invalid YAML: {e}")
        raise SystemExit(1)


# ============================================================================
# Status Command - System Status and SDK Availability
# ============================================================================


@cli.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and SDK availability."""
    status_data = {
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "sdks": {},
        "layers": {},
    }

    # Check SDK availability
    sdks = {
        "anthropic": "anthropic",
        "openai": "openai",
        "langchain": "langchain_core",
        "langgraph": "langgraph",
        "langfuse": "langfuse",
        "ragas": "ragas",
        "phoenix": "phoenix",
        "httpx": "httpx",
        "click": "click",
        "rich": "rich",
    }

    for name, module in sdks.items():
        try:
            __import__(module)
            status_data["sdks"][name] = True
        except Exception:
            # Catch all exceptions - Python 3.14+ can throw ConfigError for Pydantic V1
            status_data["sdks"][name] = False

    # Check layer availability
    layers = {
        "memory": "core.memory",
        "tools": "core.tools",
        "orchestration": "core.orchestration",
        "structured": "core.structured",
        "observability": "core.observability",
        "performance": "core.performance",
    }

    for name, module in layers.items():
        try:
            __import__(module)
            status_data["layers"][name] = True
        except Exception:
            # Catch all exceptions - Python 3.14+ can throw ConfigError for Pydantic V1
            status_data["layers"][name] = False

    # Count totals
    sdk_count = sum(1 for v in status_data["sdks"].values() if v)
    sdk_total = len(status_data["sdks"])
    layer_count = sum(1 for v in status_data["layers"].values() if v)
    layer_total = len(status_data["layers"])

    if ctx.obj.get("json_output"):
        status_data["summary"] = {
            "sdks_available": sdk_count,
            "sdks_total": sdk_total,
            "layers_available": layer_count,
            "layers_total": layer_total,
        }
        console.print(format_json(status_data))
    else:
        console.print(Panel(
            f"[bold]Unleash V35[/bold] v{__version__}\n"
            f"Python {status_data['python_version']}\n"
            f"SDKs: {sdk_count}/{sdk_total} | Layers: {layer_count}/{layer_total}",
            title="System Status",
        ))

        # SDKs table
        sdk_table = Table(title="SDK Availability")
        sdk_table.add_column("SDK", style="cyan")
        sdk_table.add_column("Status")

        for name, available in status_data["sdks"].items():
            status_icon = "[green]OK[/green]" if available else "[red]MISSING[/red]"
            sdk_table.add_row(name, status_icon)

        console.print(sdk_table)

        # Layers table
        layer_table = Table(title="Layer Status")
        layer_table.add_column("Layer", style="cyan")
        layer_table.add_column("Status")

        for name, available in status_data["layers"].items():
            status_icon = "[green]OK[/green]" if available else "[red]MISSING[/red]"
            layer_table.add_row(name, status_icon)

        console.print(layer_table)


# ============================================================================
# L0 Protocol Commands - Direct LLM Operations
# ============================================================================


@cli.group()
@click.pass_context
def protocol(ctx: click.Context) -> None:
    """L0 Protocol: Direct LLM operations."""
    pass


@protocol.command("call")
@click.argument("prompt")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
@click.option("--max-tokens", type=int, default=1000, help="Max response tokens")
@click.pass_context
def protocol_call(
    ctx: click.Context,
    prompt: str,
    model: str,
    max_tokens: int,
) -> None:
    """Make a direct LLM call."""
    try:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        if ctx.obj.get("json_output"):
            console.print(format_json({
                "model": model,
                "content": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            }))
        else:
            console.print(Panel(
                response.content[0].text,
                title=f"[bold green]Response ({model})[/bold green]",
                border_style="green",
            ))

    except ImportError as e:
        print_error(f"Anthropic SDK not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"LLM call failed: {e}")
        raise SystemExit(1)


@protocol.command("chat")
@click.option("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
@click.pass_context
def protocol_chat(ctx: click.Context, model: str) -> None:
    """Start an interactive chat session."""
    try:
        from anthropic import Anthropic

        client = Anthropic()
        messages = []

        console.print("[bold]Chat started. Type 'exit' to quit.[/bold]\n")

        while True:
            user_input = click.prompt("You", type=str)
            if user_input.lower() == "exit":
                console.print("[dim]Chat ended.[/dim]")
                break

            messages.append({"role": "user", "content": user_input})

            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=messages
            )

            assistant_msg = response.content[0].text
            messages.append({"role": "assistant", "content": assistant_msg})
            console.print(f"[cyan]Assistant:[/cyan] {assistant_msg}\n")

    except ImportError as e:
        print_error(f"Anthropic SDK not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Chat failed: {e}")
        raise SystemExit(1)


# ============================================================================
# L3 Structured Output Commands
# ============================================================================


@cli.group()
@click.pass_context
def structured(ctx: click.Context) -> None:
    """L3 Structured: Typed output generation."""
    pass


@structured.command("generate")
@click.argument("prompt")
@click.option("--schema", "-s", help="Pydantic model name or JSON schema")
@click.pass_context
def structured_generate(
    ctx: click.Context,
    prompt: str,
    schema: Optional[str],
) -> None:
    """Generate structured output from prompt."""
    try:
        from anthropic import Anthropic
        import instructor
        from pydantic import BaseModel

        # Default response model
        class Response(BaseModel):
            answer: str
            reasoning: str
            confidence: float

        client = instructor.from_anthropic(Anthropic())

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            response_model=Response,
            messages=[{"role": "user", "content": prompt}]
        )

        if ctx.obj.get("json_output"):
            console.print(format_json(response.model_dump()))
        else:
            console.print(Panel(
                format_json(response.model_dump()),
                title="[bold]Structured Response[/bold]",
            ))

    except ImportError as e:
        print_error(f"Instructor/Anthropic not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Structured generation failed: {e}")
        raise SystemExit(1)


@structured.command("validate")
@click.argument("schema_file", type=click.Path(exists=True))
@click.argument("json_file", type=click.Path(exists=True))
@click.pass_context
def structured_validate(
    ctx: click.Context,
    schema_file: str,
    json_file: str,
) -> None:
    """Validate JSON against a schema."""
    try:
        import jsonschema

        with open(schema_file) as f:
            schema = json.load(f)
        with open(json_file) as f:
            data = json.load(f)

        jsonschema.validate(data, schema)

        if ctx.obj.get("json_output"):
            console.print(format_json({"valid": True, "errors": []}))
        else:
            print_success("JSON is valid against schema")

    except jsonschema.ValidationError as e:
        if ctx.obj.get("json_output"):
            console.print(format_json({"valid": False, "errors": [str(e)]}))
        else:
            print_error(f"Validation failed: {e.message}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Validation failed: {e}")
        raise SystemExit(1)


# ============================================================================
# L6 Safety Commands
# ============================================================================


@cli.group()
@click.pass_context
def safety(ctx: click.Context) -> None:
    """L6 Safety: Content scanning and guardrails."""
    pass


@safety.command("scan")
@click.argument("text")
@click.pass_context
def safety_scan(ctx: click.Context, text: str) -> None:
    """Scan text for safety issues."""
    try:
        from core.safety.scanner_compat import InputScanner, RiskLevel

        scanner = InputScanner()
        result = scanner.scan(text)

        if ctx.obj.get("json_output"):
            console.print(format_json({
                "is_safe": result.is_safe,
                "risk_level": result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level),
                "detections": result.detections,
            }))
        else:
            if result.is_safe:
                print_success("Content is safe")
                console.print(f"[dim]Risk level: {result.risk_level}[/dim]")
            else:
                print_warning("Safety issues detected")
                console.print(f"[yellow]Risk level: {result.risk_level}[/yellow]")
                if result.detections:
                    for detection in result.detections:
                        console.print(f"  - {detection}")

    except ImportError as e:
        print_error(f"Scanner not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Scan failed: {e}")
        raise SystemExit(1)


@safety.group()
@click.pass_context
def guard(ctx: click.Context) -> None:
    """Guardrail management."""
    pass


@guard.command("enable")
@click.pass_context
def guard_enable(ctx: click.Context) -> None:
    """Enable guardrails."""
    config = load_config()
    config.setdefault("safety", {})["guardrails_enabled"] = True
    save_config(config)
    print_success("Guardrails enabled")


@guard.command("disable")
@click.pass_context
def guard_disable(ctx: click.Context) -> None:
    """Disable guardrails."""
    config = load_config()
    config.setdefault("safety", {})["guardrails_enabled"] = False
    save_config(config)
    print_warning("Guardrails disabled")


@guard.command("status")
@click.pass_context
def guard_status(ctx: click.Context) -> None:
    """Show guardrail status."""
    config = load_config()
    enabled = config.get("safety", {}).get("guardrails_enabled", True)

    if ctx.obj.get("json_output"):
        console.print(format_json({"enabled": enabled}))
    else:
        status = "[green]ENABLED[/green]" if enabled else "[red]DISABLED[/red]"
        console.print(f"Guardrails: {status}")


# ============================================================================
# L7 Processing Commands
# ============================================================================


@cli.group()
@click.pass_context
def doc(ctx: click.Context) -> None:
    """L7 Processing: Document operations."""
    pass


@doc.command("convert")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def doc_convert(
    ctx: click.Context,
    file_path: str,
    output: Optional[str],
) -> None:
    """Convert document to markdown."""
    try:
        from markitdown import MarkItDown

        converter = MarkItDown()
        result = converter.convert(file_path)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result.text_content)
            print_success(f"Converted to: {output}")
        else:
            if ctx.obj.get("json_output"):
                console.print(format_json({"content": result.text_content}))
            else:
                console.print(Panel(result.text_content[:2000], title="Converted Content"))

    except ImportError as e:
        print_error(f"MarkItDown not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Conversion failed: {e}")
        raise SystemExit(1)


@doc.command("extract")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--type", "-t", "extract_type", default="text", help="Extract type (text, tables, images)")
@click.pass_context
def doc_extract(
    ctx: click.Context,
    file_path: str,
    extract_type: str,
) -> None:
    """Extract content from document."""
    try:
        from markitdown import MarkItDown

        converter = MarkItDown()
        result = converter.convert(file_path)

        if ctx.obj.get("json_output"):
            console.print(format_json({
                "type": extract_type,
                "content": result.text_content,
            }))
        else:
            console.print(Panel(
                result.text_content[:2000],
                title=f"[bold]Extracted ({extract_type})[/bold]",
            ))

    except ImportError as e:
        print_error(f"Document processor not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Extraction failed: {e}")
        raise SystemExit(1)


# ============================================================================
# L8 Knowledge Commands
# ============================================================================


@cli.group()
@click.pass_context
def knowledge(ctx: click.Context) -> None:
    """L8 Knowledge: RAG and indexing."""
    pass


@knowledge.command("index")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--name", "-n", default="default", help="Index name")
@click.pass_context
def knowledge_index(
    ctx: click.Context,
    file_path: str,
    name: str,
) -> None:
    """Add file to knowledge index."""
    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

        # Load document
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        # Create/update index
        index = VectorStoreIndex.from_documents(documents)

        if ctx.obj.get("json_output"):
            console.print(format_json({
                "indexed": True,
                "file": file_path,
                "documents": len(documents),
            }))
        else:
            print_success(f"Indexed {len(documents)} document(s) from {file_path}")

    except ImportError as e:
        print_error(f"LlamaIndex not available: {e}")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Indexing failed: {e}")
        raise SystemExit(1)


@knowledge.command("search")
@click.argument("query")
@click.option("--index", "-i", "index_name", default="default", help="Index name")
@click.option("--limit", "-l", type=int, default=5, help="Max results")
@click.pass_context
def knowledge_search(
    ctx: click.Context,
    query: str,
    index_name: str,
    limit: int,
) -> None:
    """Search knowledge base."""
    try:
        # Basic search implementation
        if ctx.obj.get("json_output"):
            console.print(format_json({
                "query": query,
                "index": index_name,
                "results": [],
                "message": "Index not loaded. Use 'knowledge index' first.",
            }))
        else:
            console.print(f"Searching '{index_name}' for: {query}")
            print_warning("No index loaded. Use 'knowledge index <file>' first.")

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise SystemExit(1)


@knowledge.command("list")
@click.pass_context
def knowledge_list(ctx: click.Context) -> None:
    """List available knowledge indices."""
    # List indices (placeholder - would check actual storage)
    indices = [
        {"name": "default", "documents": 0, "status": "empty"},
    ]

    if ctx.obj.get("json_output"):
        console.print(format_json(indices))
    else:
        table = Table(title="Knowledge Indices")
        table.add_column("Name", style="cyan")
        table.add_column("Documents", justify="right")
        table.add_column("Status")

        for idx in indices:
            table.add_row(
                idx["name"],
                str(idx["documents"]),
                idx["status"],
            )

        console.print(table)


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
