# PHASE 8: CLI INTEGRATION & PERFORMANCE OPTIMIZATION
## Full Executable Prompt for Claude Code Implementation

---

## PREAMBLE

### Context from V33 Integration Audit
- **V33 Core**: 26/26 (100%) ✅ COMPLETE
- **Phase 5 Structured**: 32/32 (100%) ✅ COMPLETE  
- **Phase 6 Observability**: 40/43 (93%) - Python 3.14 Pydantic v1 issues (langfuse, phoenix, zep)

### User Priority
**Performance and features, full integration, NO degraded fallbacks**

### Phase 8 Objectives
1. Unified CLI interface for all V33 layers
2. Performance optimizations (pooling, caching, batching)
3. Comprehensive benchmarking suite
4. Production-ready entry points

---

## SECTION 1: CLI UNIFICATION (~400 lines)

### File: `core/cli/unified_cli.py`

Create a Click-based CLI framework integrating all V33 layers with rich output formatting.

```python
"""
Unleash Platform - Unified CLI
Click-based command-line interface with full V33 layer integration.

Requirements:
- click>=8.1.0
- rich>=13.0.0
- pyyaml>=6.0.0
- httpx>=0.27.0
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.traceback import install as install_rich_traceback
import yaml

# Install rich tracebacks globally
install_rich_traceback(show_locals=True, width=120)

console = Console()
error_console = Console(stderr=True)

T = TypeVar('T')


class OutputFormat(str, Enum):
    """Output format options."""
    TABLE = "table"
    JSON = "json"
    YAML = "yaml"
    PLAIN = "plain"


@dataclass
class CLIConfig:
    """CLI configuration loaded from .unleash.yaml or environment."""
    
    # LLM Settings
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_model: str = "claude-sonnet-4-20250514"
    
    # Memory Settings
    memory_provider: str = "mem0"
    memory_user_id: str = "default"
    
    # Observability Settings
    enable_tracing: bool = True
    trace_provider: str = "opik"
    
    # Performance Settings
    connection_pool_size: int = 10
    request_timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 3600
    
    # Output Settings
    output_format: OutputFormat = OutputFormat.TABLE
    verbose: bool = False
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CLIConfig":
        """Load config from file or environment variables."""
        config_data: dict[str, Any] = {}
        
        # Try loading from .unleash.yaml
        if config_path is None:
            config_path = Path.cwd() / ".unleash.yaml"
        
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}
        
        # Override with environment variables
        env_mapping = {
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "UNLEASH_MODEL": "default_model",
            "UNLEASH_MEMORY_PROVIDER": "memory_provider",
            "UNLEASH_MEMORY_USER_ID": "memory_user_id",
            "UNLEASH_TRACE_PROVIDER": "trace_provider",
            "UNLEASH_ENABLE_TRACING": "enable_tracing",
            "UNLEASH_POOL_SIZE": "connection_pool_size",
            "UNLEASH_TIMEOUT": "request_timeout",
            "UNLEASH_ENABLE_CACHE": "enable_caching",
            "UNLEASH_CACHE_TTL": "cache_ttl",
            "UNLEASH_FORMAT": "output_format",
            "UNLEASH_VERBOSE": "verbose",
        }
        
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Handle boolean conversion
                if config_key in ("enable_tracing", "enable_caching", "verbose"):
                    value = value.lower() in ("true", "1", "yes")
                elif config_key in ("connection_pool_size", "cache_ttl"):
                    value = int(value)
                elif config_key == "request_timeout":
                    value = float(value)
                elif config_key == "output_format":
                    value = OutputFormat(value.lower())
                config_data[config_key] = value
        
        return cls(**config_data)


# Global config reference
_cli_config: Optional[CLIConfig] = None


def get_config() -> CLIConfig:
    """Get CLI configuration, loading if necessary."""
    global _cli_config
    if _cli_config is None:
        _cli_config = CLIConfig.load()
    return _cli_config


def async_command(f: Callable[..., T]) -> Callable[..., T]:
    """Decorator to run async functions in Click commands."""
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return asyncio.run(f(*args, **kwargs))
    return wrapper


def require_sdk(sdk_name: str, package_name: str) -> Callable:
    """Decorator to require SDK availability - fails clearly, no fallback."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                __import__(package_name)
            except ImportError as e:
                error_console.print(f"[bold red]ERROR:[/bold red] SDK '{sdk_name}' is required but not available.")
                error_console.print(f"[yellow]Install with:[/yellow] pip install {package_name}")
                error_console.print(f"[dim]Import error: {e}[/dim]")
                sys.exit(1)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def format_output(data: Any, format_type: OutputFormat, title: str = "") -> None:
    """Format and display output based on selected format."""
    if format_type == OutputFormat.JSON:
        import json
        console.print(json.dumps(data, indent=2, default=str))
    elif format_type == OutputFormat.YAML:
        console.print(yaml.dump(data, default_flow_style=False))
    elif format_type == OutputFormat.PLAIN:
        console.print(str(data))
    elif format_type == OutputFormat.TABLE:
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            table = Table(title=title, show_header=True, header_style="bold magenta")
            # Add columns from first item keys
            for key in data[0].keys():
                table.add_column(str(key))
            # Add rows
            for item in data:
                table.add_row(*[str(v) for v in item.values()])
            console.print(table)
        elif isinstance(data, dict):
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for key, value in data.items():
                table.add_row(str(key), str(value))
            console.print(table)
        else:
            console.print(data)


# ============================================================================
# MAIN CLI GROUP
# ============================================================================

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml', 'plain']), 
              default='table', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.version_option(version='0.8.0', prog_name='unleash')
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], format: str, verbose: bool) -> None:
    """Unleash Platform CLI - Unified interface for all V33 layers."""
    ctx.ensure_object(dict)
    
    # Load configuration
    config_path = Path(config) if config else None
    global _cli_config
    _cli_config = CLIConfig.load(config_path)
    
    # Apply command-line overrides
    _cli_config.output_format = OutputFormat(format)
    _cli_config.verbose = verbose
    
    ctx.obj['config'] = _cli_config


# ============================================================================
# RUN COMMAND GROUP - Main execution commands
# ============================================================================

@cli.group()
def run() -> None:
    """Execute agents, pipelines, and workflows."""
    pass


@run.command('agent')
@click.argument('prompt')
@click.option('--model', '-m', default=None, help='Model to use')
@click.option('--tool', '-t', multiple=True, help='Tools to enable')
@click.option('--stream/--no-stream', default=True, help='Enable streaming')
@async_command
async def run_agent(prompt: str, model: Optional[str], tool: tuple[str, ...], stream: bool) -> None:
    """Run an agent with the specified prompt."""
    config = get_config()
    model = model or config.default_model
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running agent with {model}...", total=None)
        
        try:
            # Import core modules - fail clearly if not available
            from core.protocol.gateway import LLMGateway
            from core.tools import UnifiedToolLayer
            
            # Initialize gateway
            gateway = LLMGateway(
                default_model=model,
                pool_size=config.connection_pool_size,
                timeout=config.request_timeout,
            )
            
            # Initialize tools if specified
            tools_config = list(tool) if tool else None
            if tools_config:
                tool_layer = UnifiedToolLayer()
                await tool_layer.register_tools(tools_config)
                gateway.attach_tools(tool_layer)
            
            # Execute
            if stream:
                response_text = []
                progress.update(task, description="Streaming response...")
                async for chunk in gateway.stream(prompt):
                    response_text.append(chunk)
                    console.print(chunk, end="")
                console.print()  # Newline
            else:
                response = await gateway.complete(prompt)
                console.print(Panel(response.content, title="Response", border_style="green"))
            
        except ImportError as e:
            error_console.print(f"[bold red]ERROR:[/bold red] Required module not available: {e}")
            sys.exit(1)


@run.command('pipeline')
@click.argument('pipeline_name')
@click.option('--input', '-i', type=click.Path(exists=True), help='Input file')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--parallel/--sequential', default=True, help='Parallel execution')
@async_command
async def run_pipeline(pipeline_name: str, input: Optional[str], output: Optional[str], parallel: bool) -> None:
    """Run a named pipeline with input/output files."""
    config = get_config()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"Executing pipeline: {pipeline_name}", total=100)
        
        try:
            from core.orchestration import UnifiedOrchestrator
            
            orchestrator = UnifiedOrchestrator()
            
            # Load input data
            input_data = None
            if input:
                with open(input) as f:
                    input_data = yaml.safe_load(f) if input.endswith('.yaml') else f.read()
            
            # Execute pipeline
            progress.update(task, completed=10, description="Initializing...")
            result = await orchestrator.execute_pipeline(
                pipeline_name,
                input_data=input_data,
                parallel=parallel,
                progress_callback=lambda pct: progress.update(task, completed=int(pct * 100))
            )
            
            progress.update(task, completed=100, description="Complete!")
            
            # Save or display output
            if output:
                with open(output, 'w') as f:
                    if output.endswith('.yaml'):
                        yaml.dump(result, f)
                    else:
                        f.write(str(result))
                console.print(f"[green]Output saved to:[/green] {output}")
            else:
                format_output(result, config.output_format, title=f"Pipeline: {pipeline_name}")
                
        except ImportError as e:
            error_console.print(f"[bold red]ERROR:[/bold red] Orchestration layer not available: {e}")
            sys.exit(1)


@run.command('workflow')
@click.argument('workflow_file', type=click.Path(exists=True))
@click.option('--engine', '-e', type=click.Choice(['langgraph', 'temporal', 'crewai']), 
              default='langgraph', help='Workflow engine')
@async_command
async def run_workflow(workflow_file: str, engine: str) -> None:
    """Run a workflow from a YAML definition file."""
    config = get_config()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task(f"Running workflow with {engine}...", total=None)
        
        try:
            from core.orchestration import UnifiedOrchestrator
            
            # Load workflow definition
            with open(workflow_file) as f:
                workflow_def = yaml.safe_load(f)
            
            orchestrator = UnifiedOrchestrator()
            result = await orchestrator.execute_workflow(
                workflow_def,
                engine=engine
            )
            
            format_output(result, config.output_format, title="Workflow Result")
            
        except ImportError as e:
            error_console.print(f"[bold red]ERROR:[/bold red] Workflow engine '{engine}' not available: {e}")
            sys.exit(1)


# ============================================================================
# MEMORY COMMAND GROUP - Memory layer operations
# ============================================================================

@cli.group()
def memory() -> None:
    """Memory layer operations - store, retrieve, search."""
    pass


@memory.command('store')
@click.argument('content')
@click.option('--user', '-u', default=None, help='User ID')
@click.option('--metadata', '-m', type=str, help='JSON metadata')
@click.option('--provider', '-p', type=click.Choice(['mem0', 'letta', 'zep']), default=None)
@async_command
async def memory_store(content: str, user: Optional[str], metadata: Optional[str], provider: Optional[str]) -> None:
    """Store content in memory layer."""
    config = get_config()
    user_id = user or config.memory_user_id
    provider_name = provider or config.memory_provider
    
    try:
        from core.memory import UnifiedMemory
        import json
        
        mem = UnifiedMemory(provider=provider_name)
        meta = json.loads(metadata) if metadata else {}
        
        result = await mem.store(
            content=content,
            user_id=user_id,
            metadata=meta
        )
        
        console.print(f"[green]✓[/green] Stored memory with ID: [cyan]{result.memory_id}[/cyan]")
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Memory provider '{provider_name}' not available: {e}")
        sys.exit(1)


@memory.command('search')
@click.argument('query')
@click.option('--user', '-u', default=None, help='User ID')
@click.option('--limit', '-l', default=10, help='Max results')
@click.option('--provider', '-p', type=click.Choice(['mem0', 'letta', 'zep']), default=None)
@async_command
async def memory_search(query: str, user: Optional[str], limit: int, provider: Optional[str]) -> None:
    """Search memories by query."""
    config = get_config()
    user_id = user or config.memory_user_id
    provider_name = provider or config.memory_provider
    
    try:
        from core.memory import UnifiedMemory
        
        mem = UnifiedMemory(provider=provider_name)
        results = await mem.search(
            query=query,
            user_id=user_id,
            limit=limit
        )
        
        if results:
            table = Table(title=f"Memory Search: '{query}'", show_header=True)
            table.add_column("ID", style="cyan", width=20)
            table.add_column("Content", style="white", width=60)
            table.add_column("Score", style="green", width=10)
            
            for r in results:
                table.add_row(
                    str(r.memory_id)[:20],
                    str(r.content)[:60] + "..." if len(str(r.content)) > 60 else str(r.content),
                    f"{r.relevance_score:.3f}" if r.relevance_score else "N/A"
                )
            console.print(table)
        else:
            console.print("[yellow]No results found.[/yellow]")
            
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Memory provider '{provider_name}' not available: {e}")
        sys.exit(1)


@memory.command('list')
@click.option('--user', '-u', default=None, help='User ID')
@click.option('--limit', '-l', default=20, help='Max results')
@click.option('--provider', '-p', type=click.Choice(['mem0', 'letta', 'zep']), default=None)
@async_command
async def memory_list(user: Optional[str], limit: int, provider: Optional[str]) -> None:
    """List all memories for a user."""
    config = get_config()
    user_id = user or config.memory_user_id
    provider_name = provider or config.memory_provider
    
    try:
        from core.memory import UnifiedMemory
        
        mem = UnifiedMemory(provider=provider_name)
        memories = await mem.list(user_id=user_id, limit=limit)
        
        format_output(
            [m.__dict__ for m in memories], 
            config.output_format, 
            title=f"Memories for user: {user_id}"
        )
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Memory provider '{provider_name}' not available: {e}")
        sys.exit(1)


@memory.command('delete')
@click.argument('memory_id')
@click.option('--provider', '-p', type=click.Choice(['mem0', 'letta', 'zep']), default=None)
@click.confirmation_option(prompt='Are you sure you want to delete this memory?')
@async_command
async def memory_delete(memory_id: str, provider: Optional[str]) -> None:
    """Delete a specific memory by ID."""
    config = get_config()
    provider_name = provider or config.memory_provider
    
    try:
        from core.memory import UnifiedMemory
        
        mem = UnifiedMemory(provider=provider_name)
        await mem.delete(memory_id=memory_id)
        
        console.print(f"[green]✓[/green] Deleted memory: [cyan]{memory_id}[/cyan]")
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Memory provider '{provider_name}' not available: {e}")
        sys.exit(1)


# ============================================================================
# TOOLS COMMAND GROUP - Tool layer operations
# ============================================================================

@cli.group()
def tools() -> None:
    """Tool layer operations - list, invoke, manage."""
    pass


@tools.command('list')
@click.option('--category', '-c', type=str, help='Filter by category')
@click.option('--enabled/--all', default=True, help='Show only enabled tools')
def tools_list(category: Optional[str], enabled: bool) -> None:
    """List available tools."""
    config = get_config()
    
    try:
        from core.tools import UnifiedToolLayer
        
        tool_layer = UnifiedToolLayer()
        all_tools = tool_layer.list_tools(category=category, enabled_only=enabled)
        
        table = Table(title="Available Tools", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white", width=50)
        table.add_column("Enabled", style="green")
        
        for tool in all_tools:
            table.add_row(
                tool.name,
                tool.category,
                tool.description[:50] + "..." if len(tool.description) > 50 else tool.description,
                "✓" if tool.enabled else "✗"
            )
        
        console.print(table)
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Tool layer not available: {e}")
        sys.exit(1)


@tools.command('invoke')
@click.argument('tool_name')
@click.option('--input', '-i', type=str, help='JSON input parameters')
@click.option('--file', '-f', type=click.Path(exists=True), help='Input file (YAML/JSON)')
@async_command
async def tools_invoke(tool_name: str, input: Optional[str], file: Optional[str]) -> None:
    """Invoke a tool with specified parameters."""
    config = get_config()
    
    try:
        from core.tools import UnifiedToolLayer
        import json
        
        # Parse input
        params = {}
        if input:
            params = json.loads(input)
        elif file:
            with open(file) as f:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    params = yaml.safe_load(f)
                else:
                    params = json.load(f)
        
        tool_layer = UnifiedToolLayer()
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task(f"Invoking tool: {tool_name}...", total=None)
            result = await tool_layer.invoke(tool_name, **params)
        
        format_output(result, config.output_format, title=f"Tool Result: {tool_name}")
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Tool layer not available: {e}")
        sys.exit(1)


@tools.command('describe')
@click.argument('tool_name')
def tools_describe(tool_name: str) -> None:
    """Show detailed information about a tool."""
    try:
        from core.tools import UnifiedToolLayer
        
        tool_layer = UnifiedToolLayer()
        tool_info = tool_layer.get_tool_info(tool_name)
        
        if tool_info:
            console.print(Panel(
                f"[bold]Name:[/bold] {tool_info.name}\n"
                f"[bold]Category:[/bold] {tool_info.category}\n"
                f"[bold]Description:[/bold] {tool_info.description}\n"
                f"[bold]Parameters:[/bold]\n" + 
                "\n".join(f"  - {p.name}: {p.type} {'(required)' if p.required else '(optional)'}" 
                          for p in tool_info.parameters),
                title=f"Tool: {tool_name}",
                border_style="cyan"
            ))
        else:
            error_console.print(f"[yellow]Tool '{tool_name}' not found.[/yellow]")
            sys.exit(1)
            
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Tool layer not available: {e}")
        sys.exit(1)


# ============================================================================
# EVAL COMMAND GROUP - Evaluation operations
# ============================================================================

@cli.group()
def eval() -> None:
    """Evaluation and testing operations."""
    pass


@eval.command('run')
@click.argument('eval_file', type=click.Path(exists=True))
@click.option('--provider', '-p', type=click.Choice(['deepeval', 'ragas', 'promptfoo']), 
              default='deepeval', help='Evaluation provider')
@click.option('--output', '-o', type=click.Path(), help='Output report file')
@async_command
async def eval_run(eval_file: str, provider: str, output: Optional[str]) -> None:
    """Run evaluation suite from file."""
    config = get_config()
    
    try:
        from core.observability import ObservabilityFactory
        
        # Load eval configuration
        with open(eval_file) as f:
            eval_config = yaml.safe_load(f)
        
        evaluator = ObservabilityFactory.create_evaluator(provider)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Running evaluations...", total=len(eval_config.get('test_cases', [])))
            
            results = await evaluator.run_suite(
                eval_config,
                progress_callback=lambda: progress.advance(task)
            )
        
        # Display summary
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Test Case", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Details", style="white")
        
        for result in results:
            status_color = "green" if result.passed else "red"
            table.add_row(
                result.name,
                f"[{status_color}]{'PASS' if result.passed else 'FAIL'}[/{status_color}]",
                f"{result.score:.2f}" if result.score else "N/A",
                result.details[:40] + "..." if len(result.details) > 40 else result.details
            )
        
        console.print(table)
        
        # Summary stats
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = sum(r.score for r in results if r.score) / len([r for r in results if r.score])
        
        console.print(f"\n[bold]Summary:[/bold] {passed}/{total} passed, avg score: {avg_score:.2f}")
        
        # Save report if output specified
        if output:
            report = {
                'summary': {'passed': passed, 'total': total, 'avg_score': avg_score},
                'results': [r.__dict__ for r in results]
            }
            with open(output, 'w') as f:
                if output.endswith('.yaml'):
                    yaml.dump(report, f)
                else:
                    import json
                    json.dump(report, f, indent=2, default=str)
            console.print(f"[green]Report saved to:[/green] {output}")
            
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Evaluation provider '{provider}' not available: {e}")
        sys.exit(1)


@eval.command('metrics')
@click.option('--provider', '-p', type=click.Choice(['deepeval', 'ragas']), 
              default='deepeval', help='Evaluation provider')
def eval_metrics(provider: str) -> None:
    """List available evaluation metrics."""
    try:
        from core.observability import ObservabilityFactory
        
        evaluator = ObservabilityFactory.create_evaluator(provider)
        metrics = evaluator.list_metrics()
        
        table = Table(title=f"Available Metrics ({provider})", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white", width=50)
        
        for metric in metrics:
            table.add_row(metric.name, metric.category, metric.description)
        
        console.print(table)
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Evaluation provider '{provider}' not available: {e}")
        sys.exit(1)


# ============================================================================
# TRACE COMMAND GROUP - Tracing and observability
# ============================================================================

@cli.group()
def trace() -> None:
    """Tracing and observability operations."""
    pass


@trace.command('list')
@click.option('--limit', '-l', default=20, help='Max traces to show')
@click.option('--status', '-s', type=click.Choice(['all', 'success', 'error']), default='all')
@click.option('--provider', '-p', type=click.Choice(['opik', 'langfuse', 'phoenix']), default=None)
@async_command
async def trace_list(limit: int, status: str, provider: Optional[str]) -> None:
    """List recent traces."""
    config = get_config()
    provider_name = provider or config.trace_provider
    
    try:
        from core.observability import ObservabilityFactory
        
        tracer = ObservabilityFactory.create_tracer(provider_name)
        traces = await tracer.list_traces(limit=limit, status=status if status != 'all' else None)
        
        table = Table(title="Recent Traces", show_header=True)
        table.add_column("Trace ID", style="cyan", width=24)
        table.add_column("Name", style="white", width=30)
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Timestamp", style="dim")
        
        for trace in traces:
            status_color = "green" if trace.status == "success" else "red"
            table.add_row(
                trace.trace_id[:24],
                trace.name[:30],
                f"[{status_color}]{trace.status}[/{status_color}]",
                f"{trace.duration_ms:.1f}ms" if trace.duration_ms else "N/A",
                trace.timestamp.strftime("%Y-%m-%d %H:%M") if trace.timestamp else "N/A"
            )
        
        console.print(table)
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Trace provider '{provider_name}' not available: {e}")
        sys.exit(1)


@trace.command('show')
@click.argument('trace_id')
@click.option('--provider', '-p', type=click.Choice(['opik', 'langfuse', 'phoenix']), default=None)
@async_command
async def trace_show(trace_id: str, provider: Optional[str]) -> None:
    """Show detailed trace information."""
    config = get_config()
    provider_name = provider or config.trace_provider
    
    try:
        from core.observability import ObservabilityFactory
        
        tracer = ObservabilityFactory.create_tracer(provider_name)
        trace = await tracer.get_trace(trace_id)
        
        if trace:
            console.print(Panel(
                f"[bold]Trace ID:[/bold] {trace.trace_id}\n"
                f"[bold]Name:[/bold] {trace.name}\n"
                f"[bold]Status:[/bold] {trace.status}\n"
                f"[bold]Duration:[/bold] {trace.duration_ms:.2f}ms\n"
                f"[bold]Timestamp:[/bold] {trace.timestamp}\n"
                f"[bold]Spans:[/bold] {len(trace.spans)}",
                title="Trace Details",
                border_style="cyan"
            ))
            
            # Show spans
            if trace.spans:
                console.print("\n[bold]Spans:[/bold]")
                for span in trace.spans:
                    console.print(f"  └─ {span.name} ({span.duration_ms:.1f}ms) [{span.status}]")
        else:
            error_console.print(f"[yellow]Trace '{trace_id}' not found.[/yellow]")
            
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Trace provider '{provider_name}' not available: {e}")
        sys.exit(1)


@trace.command('export')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output file')
@click.option('--format', '-f', type=click.Choice(['json', 'otlp']), default='json', help='Export format')
@click.option('--limit', '-l', default=100, help='Max traces to export')
@click.option('--provider', '-p', type=click.Choice(['opik', 'langfuse', 'phoenix']), default=None)
@async_command  
async def trace_export(output: str, format: str, limit: int, provider: Optional[str]) -> None:
    """Export traces to file."""
    config = get_config()
    provider_name = provider or config.trace_provider
    
    try:
        from core.observability import ObservabilityFactory
        import json
        
        tracer = ObservabilityFactory.create_tracer(provider_name)
        traces = await tracer.list_traces(limit=limit)
        
        if format == 'json':
            export_data = [t.__dict__ for t in traces]
            with open(output, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == 'otlp':
            # Export in OpenTelemetry format
            otlp_data = tracer.export_otlp(traces)
            with open(output, 'wb') as f:
                f.write(otlp_data)
        
        console.print(f"[green]✓[/green] Exported {len(traces)} traces to: [cyan]{output}[/cyan]")
        
    except ImportError as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Trace provider '{provider_name}' not available: {e}")
        sys.exit(1)


# ============================================================================
# CONFIG COMMAND GROUP - Configuration management
# ============================================================================

@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command('show')
def config_show() -> None:
    """Show current configuration."""
    cfg = get_config()
    
    table = Table(title="Current Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")
    
    for field_name in cfg.__dataclass_fields__:
        value = getattr(cfg, field_name)
        # Mask sensitive values
        if 'api_key' in field_name.lower() and value:
            display_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "****"
        else:
            display_value = str(value)
        table.add_row(field_name, display_value, "env/file")
    
    console.print(table)


@config.command('init')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing config')
def config_init(force: bool) -> None:
    """Initialize .unleash.yaml configuration file."""
    config_path = Path.cwd() / ".unleash.yaml"
    
    if config_path.exists() and not force:
        error_console.print("[yellow]Config file already exists. Use --force to overwrite.[/yellow]")
        sys.exit(1)
    
    default_config = {
        'unleash': {
            'version': '0.8.0',
            'model': 'claude-sonnet-4-20250514',
            'memory': {
                'provider': 'mem0',
                'user_id': 'default'
            },
            'observability': {
                'enabled': True,
                'provider': 'opik'
            },
            'performance': {
                'pool_size': 10,
                'timeout': 30.0,
                'cache_enabled': True,
                'cache_ttl': 3600
            },
            'output': {
                'format': 'table',
                'verbose': False
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[green]✓[/green] Created configuration file: [cyan]{config_path}[/cyan]")


@config.command('validate')
def config_validate() -> None:
    """Validate current configuration."""
    try:
        cfg = CLIConfig.load()
        
        issues = []
        
        # Check required API keys
        if not cfg.anthropic_api_key and not cfg.openai_api_key:
            issues.append("No API key configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        
        # Validate settings
        if cfg.connection_pool_size < 1:
            issues.append("connection_pool_size must be >= 1")
        if cfg.request_timeout < 1:
            issues.append("request_timeout must be >= 1")
        if cfg.cache_ttl < 0:
            issues.append("cache_ttl must be >= 0")
        
        if issues:
            console.print("[yellow]Configuration issues found:[/yellow]")
            for issue in issues:
                console.print(f"  [red]✗[/red] {issue}")
            sys.exit(1)
        else:
            console.print("[green]✓[/green] Configuration is valid")
            
    except Exception as e:
        error_console.print(f"[bold red]ERROR:[/bold red] Configuration validation failed: {e}")
        sys.exit(1)


# ============================================================================
# STATUS COMMAND - System status
# ============================================================================

@cli.command('status')
def status() -> None:
    """Show system status and SDK availability."""
    console.print(Panel("[bold]Unleash Platform Status[/bold]", border_style="blue"))
    
    # SDK availability table
    table = Table(title="SDK Availability", show_header=True)
    table.add_column("Layer", style="magenta")
    table.add_column("SDK", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes", style="dim")
    
    # Check each SDK
    sdks = [
        ("Protocol", "anthropic", "anthropic"),
        ("Protocol", "openai", "openai"),
        ("Protocol", "litellm", "litellm"),
        ("Memory", "mem0ai", "mem0ai"),
        ("Memory", "letta", "letta"),
        ("Memory", "zep-python", "zep_python"),
        ("Orchestration", "langgraph", "langgraph"),
        ("Orchestration", "temporalio", "temporalio"),
        ("Orchestration", "crewai", "crewai"),
        ("Structured", "instructor", "instructor"),
        ("Structured", "pydantic-ai", "pydantic_ai"),
        ("Observability", "opik", "opik"),
        ("Observability", "deepeval", "deepeval"),
        ("Observability", "langfuse", "langfuse"),
        ("Observability", "phoenix", "phoenix"),
    ]
    
    for layer, name, package in sdks:
        try:
            __import__(package)
            table.add_row(layer, name, "[green]✓ Available[/green]", "")
        except ImportError as e:
            table.add_row(layer, name, "[red]✗ Missing[/red]", str(e)[:30])
    
    console.print(table)
    
    # Config status
    cfg = get_config()
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Model: {cfg.default_model}")
    console.print(f"  Memory Provider: {cfg.memory_provider}")
    console.print(f"  Trace Provider: {cfg.trace_provider}")
    console.print(f"  Caching: {'Enabled' if cfg.enable_caching else 'Disabled'}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
```

---

## SECTION 2: PERFORMANCE OPTIMIZATIONS (~300 lines)

### File: `core/performance/optimizer.py`

Create performance optimization layer with connection pooling, caching, and batching.

```python
"""
Unleash Platform - Performance Optimizer
Connection pooling, caching, batching, and profiling utilities.

Requirements:
- httpx>=0.27.0
- cachetools>=5.3.0
- redis (optional)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union
from datetime import datetime, timedelta
import logging

import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ============================================================================
# CONNECTION POOLING
# ============================================================================

@dataclass
class PoolConfig:
    """Configuration for connection pools."""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0


class HTTPConnectionPool:
    """Managed HTTP connection pool with provider-specific clients."""
    
    _instance: Optional["HTTPConnectionPool"] = None
    _lock: asyncio.Lock
    
    def __new__(cls) -> "HTTPConnectionPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()
        self._config = PoolConfig()
        self._initialized = True
    
    def configure(self, config: PoolConfig) -> None:
        """Update pool configuration."""
        self._config = config
    
    async def get_client(
        self, 
        provider: str,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> httpx.AsyncClient:
        """Get or create a client for the specified provider."""
        async with self._lock:
            if provider not in self._clients:
                limits = httpx.Limits(
                    max_connections=self._config.max_connections,
                    max_keepalive_connections=self._config.max_keepalive_connections,
                    keepalive_expiry=self._config.keepalive_expiry
                )
                timeout = httpx.Timeout(
                    connect=self._config.connect_timeout,
                    read=self._config.read_timeout,
                    write=self._config.write_timeout,
                    pool=self._config.pool_timeout
                )
                
                self._clients[provider] = httpx.AsyncClient(
                    base_url=base_url,
                    headers=headers or {},
                    limits=limits,
                    timeout=timeout,
                    http2=True  # Enable HTTP/2 for connection multiplexing
                )
                logger.info(f"Created connection pool for provider: {provider}")
            
            return self._clients[provider]
    
    async def close_all(self) -> None:
        """Close all connection pools."""
        async with self._lock:
            for provider, client in self._clients.items():
                await client.aclose()
                logger.info(f"Closed connection pool for provider: {provider}")
            self._clients.clear()
    
    async def close_provider(self, provider: str) -> None:
        """Close connection pool for a specific provider."""
        async with self._lock:
            if provider in self._clients:
                await self._clients[provider].aclose()
                del self._clients[provider]
                logger.info(f"Closed connection pool for provider: {provider}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            provider: {
                'is_closed': client.is_closed,
            }
            for provider, client in self._clients.items()
        }


# Singleton accessor
def get_pool() -> HTTPConnectionPool:
    """Get the global connection pool instance."""
    return HTTPConnectionPool()


# ============================================================================
# CACHING LAYER
# ============================================================================

@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with TTL tracking."""
    value: V
    expires_at: float
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 1000, ttl: float = 3600.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: K) -> Optional[V]:
        """Get value from cache if exists and not expired."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry.expires_at:
                    entry.access_count += 1
                    self._hits += 1
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    return entry.value
                else:
                    # Expired
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional custom TTL."""
        async with self._lock:
            ttl = ttl or self._ttl
            expires_at = time.time() + ttl
            
            # Remove oldest if at capacity
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
            self._cache.move_to_end(key)
    
    async def invalidate(self, key: K) -> bool:
        """Remove key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'size': len(self._cache),
            'maxsize': self._maxsize,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'ttl': self._ttl
        }


class RedisCache(Generic[K, V]):
    """Redis-backed cache for distributed caching (optional)."""
    
    def __init__(
        self, 
        url: str = "redis://localhost:6379",
        prefix: str = "unleash:",
        ttl: float = 3600.0
    ) -> None:
        self._url = url
        self._prefix = prefix
        self._ttl = ttl
        self._redis: Optional[Any] = None
    
    async def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self._url)
            except ImportError:
                raise ImportError("redis package required for RedisCache: pip install redis")
        return self._redis
    
    def _make_key(self, key: K) -> str:
        """Create prefixed cache key."""
        key_str = str(key) if not isinstance(key, str) else key
        return f"{self._prefix}{key_str}"
    
    async def get(self, key: K) -> Optional[V]:
        """Get value from Redis."""
        import json
        client = await self._get_client()
        value = await client.get(self._make_key(key))
        if value is not None:
            return json.loads(value)
        return None
    
    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set value in Redis with TTL."""
        import json
        client = await self._get_client()
        ttl_seconds = int(ttl or self._ttl)
        await client.setex(
            self._make_key(key),
            ttl_seconds,
            json.dumps(value, default=str)
        )
    
    async def invalidate(self, key: K) -> bool:
        """Remove key from Redis."""
        client = await self._get_client()
        result = await client.delete(self._make_key(key))
        return result > 0
    
    async def clear(self, pattern: str = "*") -> None:
        """Clear keys matching pattern."""
        client = await self._get_client()
        keys = await client.keys(f"{self._prefix}{pattern}")
        if keys:
            await client.delete(*keys)


class CacheManager:
    """Unified cache manager with fallback support."""
    
    def __init__(
        self,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
        maxsize: int = 1000,
        ttl: float = 3600.0
    ) -> None:
        self._ttl = ttl
        self._lru = LRUCache[str, Any](maxsize=maxsize, ttl=ttl)
        self._redis: Optional[RedisCache[str, Any]] = None
        
        if use_redis:
            try:
                self._redis = RedisCache(url=redis_url, ttl=ttl)
            except ImportError:
                logger.warning("Redis not available, using LRU cache only")
    
    def _make_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        key_data = f"{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with Redis fallback."""
        # Try LRU first (faster)
        value = await self._lru.get(key)
        if value is not None:
            return value
        
        # Try Redis if available
        if self._redis:
            value = await self._redis.get(key)
            if value is not None:
                # Populate LRU cache
                await self._lru.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set in all cache layers."""
        await self._lru.set(key, value, ttl)
        if self._redis:
            await self._redis.set(key, value, ttl)
    
    def cached(
        self, 
        ttl: Optional[float] = None,
        key_prefix: str = ""
    ) -> Callable:
        """Decorator for caching async function results."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                cache_key = f"{key_prefix}:{func.__name__}:{self._make_cache_key(*args, **kwargs)}"
                
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get unified cache statistics."""
        return {
            'lru': self._lru.get_stats(),
            'redis_enabled': self._redis is not None
        }


# ============================================================================
# REQUEST DEDUPLICATION
# ============================================================================

class RequestDeduplicator:
    """Deduplicates concurrent identical requests."""
    
    def __init__(self) -> None:
        self._pending: Dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
    
    def _make_request_key(self, func_name: str, *args: Any, **kwargs: Any) -> str:
        """Generate unique request key."""
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    async def execute(
        self,
        key: str,
        coro: Callable[[], Any]
    ) -> Any:
        """Execute coroutine, deduplicating concurrent identical requests."""
        async with self._lock:
            if key in self._pending:
                # Wait for existing request
                logger.debug(f"Deduplicating request: {key[:16]}...")
                return await self._pending[key]
            
            # Create new future
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            self._pending[key] = future
        
        try:
            result = await coro()
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._lock:
                self._pending.pop(key, None)
    
    def dedupe(self) -> Callable:
        """Decorator for request deduplication."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                key = self._make_request_key(func.__name__, *args, **kwargs)
                return await self.execute(key, lambda: func(*args, **kwargs))
            return wrapper
        return decorator


# ============================================================================
# ASYNC BATCH PROCESSOR
# ============================================================================

@dataclass
class BatchItem(Generic[T]):
    """Item in a batch queue."""
    data: T
    future: asyncio.Future[Any]
    added_at: float = field(default_factory=time.time)


class BatchProcessor(Generic[T]):
    """Batches requests for efficient processing."""
    
    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
        processor: Optional[Callable[[List[T]], List[Any]]] = None
    ) -> None:
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._processor = processor
        self._queue: List[BatchItem[T]] = []
        self._lock = asyncio.Lock()
        self._process_task: Optional[asyncio.Task[None]] = None
    
    async def add(self, item: T) -> Any:
        """Add item to batch and wait for result."""
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._queue.append(BatchItem(data=item, future=future))
            
            # Start processor if not running
            if self._process_task is None or self._process_task.done():
                self._process_task = asyncio.create_task(self._process_loop())
        
        return await future
    
    async def _process_loop(self) -> None:
        """Process batches."""
        while True:
            await asyncio.sleep(self._batch_timeout)
            
            async with self._lock:
                if not self._queue:
                    return
                
                # Take batch
                batch = self._queue[:self._batch_size]
                self._queue = self._queue[self._batch_size:]
            
            if batch and self._processor:
                try:
                    items = [b.data for b in batch]
                    results = await self._processor(items)
                    
                    for item, result in zip(batch, results):
                        if not item.future.done():
                            item.future.set_result(result)
                            
                except Exception as e:
                    for item in batch:
                        if not item.future.done():
                            item.future.set_exception(e)


# ============================================================================
# LAZY SDK LOADING
# ============================================================================

class LazyLoader:
    """Lazy loader for SDK modules."""
    
    _loaded: Dict[str, Any] = {}
    _lock: asyncio.Lock = asyncio.Lock()
    
    @classmethod
    async def load(cls, module_name: str) -> Any:
        """Load module lazily."""
        async with cls._lock:
            if module_name not in cls._loaded:
                import importlib
                try:
                    module = importlib.import_module(module_name)
                    cls._loaded[module_name] = module
                    logger.info(f"Lazily loaded module: {module_name}")
                except ImportError as e:
                    logger.error(f"Failed to load module {module_name}: {e}")
                    raise
            return cls._loaded[module_name]
    
    @classmethod
    def is_loaded(cls, module_name: str) -> bool:
        """Check if module is loaded."""
        return module_name in cls._loaded
    
    @classmethod
    def unload(cls, module_name: str) -> None:
        """Unload module."""
        cls._loaded.pop(module_name, None)


def lazy_import(module_name: str) -> Callable:
    """Decorator for lazy SDK imports."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            await LazyLoader.load(module_name)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# PROFILING HOOKS
# ============================================================================

@dataclass
class TimingRecord:
    """Record of a timed operation."""
    name: str
    duration_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Performance profiling with timing analysis."""
    
    def __init__(self, max_records: int = 10000) -> None:
        self._records: List[TimingRecord] = []
        self._max_records = max_records
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def measure(self, name: str, **metadata: Any):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            await self._add_record(TimingRecord(
                name=name,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                metadata=metadata
            ))
    
    async def _add_record(self, record: TimingRecord) -> None:
        """Add timing record."""
        async with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
    
    def timed(self, name: Optional[str] = None) -> Callable:
        """Decorator for timing async functions."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            operation_name = name or func.__name__
            
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                async with self.measure(operation_name):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        async with self._lock:
            records = [r for r in self._records if name is None or r.name == name]
        
        if not records:
            return {'count': 0}
        
        durations = [r.duration_ms for r in records]
        return {
            'count': len(records),
            'total_ms': sum(durations),
            'avg_ms': sum(durations) / len(durations),
            'min_ms': min(durations),
            'max_ms': max(durations),
            'p50_ms': sorted(durations)[len(durations) // 2],
            'p99_ms': sorted(durations)[int(len(durations) * 0.99)] if len(durations) >= 100 else max(durations)
        }
    
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operation names."""
        names = set(r.name for r in self._records)
        return {name: await self.get_stats(name) for name in names}


# ============================================================================
# UNIFIED PERFORMANCE MANAGER
# ============================================================================

class PerformanceManager:
    """Unified performance optimization manager."""
    
    _instance: Optional["PerformanceManager"] = None
    
    def __new__(cls) -> "PerformanceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self.pool = HTTPConnectionPool()
        self.cache = CacheManager(use_redis=False, maxsize=1000, ttl=3600)
        self.deduplicator = RequestDeduplicator()
        self.profiler = Profiler()
        self._initialized = True
    
    def configure(
        self,
        pool_config: Optional[PoolConfig] = None,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
        cache_maxsize: int = 1000,
        cache_ttl: float = 3600.0
    ) -> None:
        """Configure performance manager."""
        if pool_config:
            self.pool.configure(pool_config)
        
        self.cache = CacheManager(
            use_redis=use_redis,
            redis_url=redis_url,
            maxsize=cache_maxsize,
            ttl=cache_ttl
        )
    
    async def get_client(self, provider: str, **kwargs: Any) -> httpx.AsyncClient:
        """Get pooled HTTP client."""
        return await self.pool.get_client(provider, **kwargs)
    
    async def shutdown(self) -> None:
        """Shutdown all resources."""
        await self.pool.close_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all performance statistics."""
        return {
            'pool': self.pool.get_stats(),
            'cache': self.cache.get_stats(),
        }


# Singleton accessor
def get_performance_manager() -> PerformanceManager:
    """Get the global performance manager."""
    return PerformanceManager()
```

---

## SECTION 3: ENTRY POINT INTEGRATION (~200 lines)

### File: `cli.py` (root level)

Create the main entry point with auto-discovery and error handling.

```python
#!/usr/bin/env python
"""
Unleash Platform - Main CLI Entry Point

This is the primary entry point for the Unleash CLI.
Run with: python cli.py [command] or unleash [command]

Requirements:
- click>=8.1.0
- rich>=13.0.0
- pyyaml>=6.0.0
"""

from __future__ import annotations

import sys
import os
import importlib
import pkgutil
from pathlib import Path
from typing import List, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.panel import Panel

# Install rich tracebacks for better error display
install_rich_traceback(
    show_locals=True,
    width=120,
    extra_lines=3,
    theme="monokai"
)

console = Console()
error_console = Console(stderr=True)

# Version info
__version__ = "0.8.0"
__author__ = "Unleash Platform"


def get_version_info() -> str:
    """Get detailed version information."""
    import platform
    
    python_version = platform.python_version()
    system = platform.system()
    machine = platform.machine()
    
    # Check key SDK versions
    sdk_versions = []
    for sdk, package in [("anthropic", "anthropic"), ("click", "click"), ("rich", "rich")]:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")
            sdk_versions.append(f"  {sdk}: {version}")
        except ImportError:
            sdk_versions.append(f"  {sdk}: not installed")
    
    return f"""Unleash Platform CLI v{__version__}

Python: {python_version}
Platform: {system} ({machine})

Core SDKs:
{chr(10).join(sdk_versions)}
"""


def discover_commands() -> List[str]:
    """Auto-discover available command groups."""
    commands = []
    cli_path = PROJECT_ROOT / "core" / "cli"
    
    if cli_path.exists():
        for module_info in pkgutil.iter_modules([str(cli_path)]):
            if not module_info.name.startswith("_"):
                commands.append(module_info.name)
    
    return commands


def setup_environment() -> None:
    """Setup environment for CLI execution."""
    # Load .env if exists
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass  # dotenv not required
    
    # Set default environment variables
    os.environ.setdefault("UNLEASH_LOG_LEVEL", "INFO")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def handle_keyboard_interrupt() -> None:
    """Handle Ctrl+C gracefully."""
    console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    sys.exit(130)  # Standard exit code for SIGINT


def handle_exception(exc: Exception) -> None:
    """Handle uncaught exceptions with rich formatting."""
    error_console.print(Panel(
        f"[bold red]Error:[/bold red] {type(exc).__name__}\n\n"
        f"{str(exc)}\n\n"
        f"[dim]Use --verbose for full traceback[/dim]",
        title="Unleash CLI Error",
        border_style="red"
    ))
    sys.exit(1)


@click.group(invoke_without_command=True)
@click.option("--version", "-V", is_flag=True, help="Show version and exit")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def main(ctx: click.Context, version: bool, verbose: bool, debug: bool) -> None:
    """
    Unleash Platform CLI - Unified interface for AI agent development.
    
    Run 'unleash COMMAND --help' for more information on a command.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    
    if version:
        console.print(get_version_info())
        sys.exit(0)
    
    if ctx.invoked_subcommand is None:
        # Show help if no command provided
        console.print(Panel(
            "[bold]Unleash Platform CLI[/bold]\n\n"
            "A unified interface for building and deploying AI agents.\n\n"
            "[bold]Quick Start:[/bold]\n"
            "  unleash run agent \"Hello, world!\"\n"
            "  unleash memory search \"previous conversation\"\n"
            "  unleash tools list\n"
            "  unleash eval run tests.yaml\n"
            "  unleash trace list\n\n"
            "[dim]Run 'unleash --help' for all commands[/dim]",
            title="Welcome",
            border_style="blue"
        ))


# ============================================================================
# CORE COMMANDS (loaded from core.cli.unified_cli)
# ============================================================================

def load_cli_commands() -> None:
    """Load all CLI commands from unified_cli module."""
    try:
        from core.cli.unified_cli import (
            run, memory, tools, eval, trace, 
            config, status
        )
        
        # Add command groups
        main.add_command(run)
        main.add_command(memory)
        main.add_command(tools)
        main.add_command(eval)
        main.add_command(trace)
        main.add_command(config)
        main.add_command(status)
        
    except ImportError as e:
        # Create placeholder commands that show error
        @main.command()
        def run():
            """Run commands (not available)"""
            error_console.print(f"[red]CLI module not available:[/red] {e}")
            sys.exit(1)


# ============================================================================
# ADDITIONAL ROOT COMMANDS
# ============================================================================

@main.command()
@click.option("--all", "-a", is_flag=True, help="Run all checks")
def doctor(all: bool) -> None:
    """Check system health and SDK availability."""
    console.print("[bold]Running system diagnostics...[/bold]\n")
    
    checks = [
        ("Python version", check_python_version),
        ("Required packages", check_required_packages),
        ("API keys", check_api_keys),
        ("Core modules", check_core_modules),
    ]
    
    if all:
        checks.extend([
            ("Optional SDKs", check_optional_sdks),
            ("Connection pooling", check_connection_pooling),
        ])
    
    all_passed = True
    for name, check_func in checks:
        try:
            passed, message = check_func()
            status = "[green]✓[/green]" if passed else "[red]✗[/red]"
            console.print(f"  {status} {name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            console.print(f"  [red]✗[/red] {name}: Error - {e}")
            all_passed = False
    
    console.print()
    if all_passed:
        console.print("[green]All checks passed![/green]")
    else:
        console.print("[yellow]Some checks failed. Review above for details.[/yellow]")
        sys.exit(1)


def check_python_version() -> tuple[bool, str]:
    """Check Python version compatibility."""
    import platform
    version = platform.python_version_tuple()
    major, minor = int(version[0]), int(version[1])
    
    if major >= 3 and minor >= 11:
        return True, f"{platform.python_version()}"
    return False, f"{platform.python_version()} (requires >= 3.11)"


def check_required_packages() -> tuple[bool, str]:
    """Check required packages are installed."""
    required = ["click", "rich", "pyyaml", "httpx"]
    missing = []
    
    for package in required:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All installed"


def check_api_keys() -> tuple[bool, str]:
    """Check API keys are configured."""
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    found = [k for k in keys if os.environ.get(k)]
    
    if found:
        return True, f"Found: {', '.join(found)}"
    return False, "No API keys configured"


def check_core_modules() -> tuple[bool, str]:
    """Check core modules can be imported."""
    modules = [
        "core.cli.unified_cli",
        "core.performance.optimizer",
    ]
    errors = []
    
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            errors.append(f"{module}: {e}")
    
    if errors:
        return False, f"Import errors: {len(errors)}"
    return True, "All modules loadable"


def check_optional_sdks() -> tuple[bool, str]:
    """Check optional SDKs."""
    sdks = ["anthropic", "openai", "litellm", "mem0ai", "instructor"]
    available = []
    
    for sdk in sdks:
        try:
            importlib.import_module(sdk)
            available.append(sdk)
        except ImportError:
            pass
    
    return True, f"{len(available)}/{len(sdks)} available"


def check_connection_pooling() -> tuple[bool, str]:
    """Check connection pooling."""
    try:
        import httpx
        return True, f"httpx {httpx.__version__}"
    except ImportError:
        return False, "httpx not installed"


@main.command()
def info() -> None:
    """Show detailed system information."""
    console.print(get_version_info())
    
    # Show discovered commands
    commands = discover_commands()
    if commands:
        console.print(f"\nDiscovered command modules: {', '.join(commands)}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def cli_main() -> None:
    """Main entry point with error handling."""
    setup_environment()
    
    try:
        # Load CLI commands
        load_cli_commands()
        
        # Run CLI
        main()
        
    except KeyboardInterrupt:
        handle_keyboard_interrupt()
    except click.ClickException:
        raise  # Let Click handle its own exceptions
    except Exception as exc:
        if os.environ.get("UNLEASH_DEBUG") == "1":
            raise  # Re-raise for debugging
        handle_exception(exc)


if __name__ == "__main__":
    cli_main()
```

---

## SECTION 4: PERFORMANCE BENCHMARKS (~200 lines)

### File: `scripts/benchmark_performance.py`

Create comprehensive benchmarking suite for all layers.

```python
#!/usr/bin/env python
"""
Unleash Platform - Performance Benchmarks

Benchmarks for:
- Memory layer throughput (ops/sec)
- Tool layer latency (ms)
- Orchestration startup time
- Structured output parsing speed
- End-to-end pipeline benchmark

Run with: python scripts/benchmark_performance.py
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    ops_per_sec: float
    memory_mb: float = 0.0
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'iterations': self.iterations,
            'total_ms': round(self.total_time_ms, 3),
            'avg_ms': round(self.avg_time_ms, 3),
            'min_ms': round(self.min_time_ms, 3),
            'max_ms': round(self.max_time_ms, 3),
            'std_dev_ms': round(self.std_dev_ms, 3),
            'ops_sec': round(self.ops_per_sec, 1),
            'memory_mb': round(self.memory_mb, 2)
        }


class Benchmarker:
    """Benchmark runner with timing and memory tracking."""
    
    def __init__(self, warmup_iterations: int = 5) -> None:
        self.warmup_iterations = warmup_iterations
        self.results: List[BenchmarkResult] = []
    
    async def run_async(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 100
    ) -> BenchmarkResult:
        """Run async benchmark."""
        # Warmup
        for _ in range(self.warmup_iterations):
            await func()
        
        # Force GC before measurement
        gc.collect()
        
        # Measure
        times: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            await func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        # Calculate stats
        total_time = sum(times)
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=std_dev,
            ops_per_sec=(iterations / total_time) * 1000 if total_time > 0 else 0
        )
        
        self.results.append(result)
        return result
    
    def run_sync(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 100
    ) -> BenchmarkResult:
        """Run sync benchmark."""
        # Warmup
        for _ in range(self.warmup_iterations):
            func()
        
        gc.collect()
        
        times: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=std_dev,
            ops_per_sec=(iterations / total_time) * 1000 if total_time > 0 else 0
        )
        
        self.results.append(result)
        return result
    
    def print_results(self) -> None:
        """Print results as rich table."""
        table = Table(title="Performance Benchmark Results", show_header=True)
        table.add_column("Benchmark", style="cyan")
        table.add_column("Iterations", justify="right")
        table.add_column("Avg (ms)", justify="right", style="yellow")
        table.add_column("Min (ms)", justify="right", style="green")
        table.add_column("Max (ms)", justify="right", style="red")
        table.add_column("Std Dev", justify="right")
        table.add_column("Ops/sec", justify="right", style="magenta")
        
        for r in self.results:
            table.add_row(
                r.name,
                str(r.iterations),
                f"{r.avg_time_ms:.3f}",
                f"{r.min_time_ms:.3f}",
                f"{r.max_time_ms:.3f}",
                f"{r.std_dev_ms:.3f}",
                f"{r.ops_per_sec:.1f}"
            )
        
        console.print(table)


# ============================================================================
# BENCHMARK IMPLEMENTATIONS
# ============================================================================

async def benchmark_memory_layer(bench: Benchmarker) -> None:
    """Benchmark memory layer operations."""
    console.print("\n[bold]Memory Layer Benchmarks[/bold]")
    
    try:
        from core.memory import UnifiedMemory
        
        mem = UnifiedMemory(provider="mem0")
        test_content = "This is a test memory for benchmarking purposes."
        
        # Store benchmark
        async def store_op():
            await mem.store(content=test_content, user_id="benchmark_user")
        
        await bench.run_async("Memory: Store", store_op, iterations=50)
        
        # Search benchmark
        async def search_op():
            await mem.search(query="test memory", user_id="benchmark_user", limit=5)
        
        await bench.run_async("Memory: Search", search_op, iterations=50)
        
        # List benchmark
        async def list_op():
            await mem.list(user_id="benchmark_user", limit=10)
        
        await bench.run_async("Memory: List", list_op, iterations=50)
        
    except ImportError as e:
        console.print(f"[yellow]Skipping: Memory layer not available ({e})[/yellow]")


async def benchmark_tool_layer(bench: Benchmarker) -> None:
    """Benchmark tool layer operations."""
    console.print("\n[bold]Tool Layer Benchmarks[/bold]")
    
    try:
        from core.tools import UnifiedToolLayer
        
        tool_layer = UnifiedToolLayer()
        
        # List tools benchmark
        def list_tools_op():
            tool_layer.list_tools()
        
        bench.run_sync("Tools: List All", list_tools_op, iterations=100)
        
        # Get tool info benchmark
        def get_info_op():
            tool_layer.get_tool_info("web_search")
        
        bench.run_sync("Tools: Get Info", get_info_op, iterations=100)
        
        # Tool invocation (mock - no actual API call)
        async def invoke_mock():
            # Simulate tool preparation without actual call
            pass
        
        await bench.run_async("Tools: Invoke Prep", invoke_mock, iterations=100)
        
    except ImportError as e:
        console.print(f"[yellow]Skipping: Tool layer not available ({e})[/yellow]")


async def benchmark_orchestration_startup(bench: Benchmarker) -> None:
    """Benchmark orchestration layer startup."""
    console.print("\n[bold]Orchestration Layer Benchmarks[/bold]")
    
    try:
        from core.orchestration import UnifiedOrchestrator
        
        # Startup time benchmark
        def startup_op():
            orchestrator = UnifiedOrchestrator()
            return orchestrator
        
        bench.run_sync("Orchestration: Startup", startup_op, iterations=20)
        
    except ImportError as e:
        console.print(f"[yellow]Skipping: Orchestration layer not available ({e})[/yellow]")


async def benchmark_structured_output(bench: Benchmarker) -> None:
    """Benchmark structured output parsing."""
    console.print("\n[bold]Structured Output Benchmarks[/bold]")
    
    try:
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            name: str
            value: int
            tags: List[str]
        
        test_data = {"name": "test", "value": 42, "tags": ["a", "b", "c"]}
        
        # Pydantic parsing benchmark
        def parse_op():
            TestModel(**test_data)
        
        bench.run_sync("Structured: Pydantic Parse", parse_op, iterations=1000)
        
        # JSON validation benchmark
        import json
        json_str = json.dumps(test_data)
        
        def json_parse_op():
            data = json.loads(json_str)
            TestModel(**data)
        
        bench.run_sync("Structured: JSON + Parse", json_parse_op, iterations=1000)
        
    except ImportError as e:
        console.print(f"[yellow]Skipping: Structured output not available ({e})[/yellow]")


async def benchmark_caching(bench: Benchmarker) -> None:
    """Benchmark caching layer."""
    console.print("\n[bold]Cache Layer Benchmarks[/bold]")
    
    try:
        from core.performance.optimizer import CacheManager
        
        cache = CacheManager(use_redis=False, maxsize=10000)
        
        # Cache set benchmark
        async def cache_set_op():
            await cache.set("benchmark_key", {"data": "value"})
        
        await bench.run_async("Cache: Set", cache_set_op, iterations=1000)
        
        # Cache get (hit) benchmark
        await cache.set("hit_key", {"data": "cached"})
        
        async def cache_get_hit_op():
            await cache.get("hit_key")
        
        await bench.run_async("Cache: Get (hit)", cache_get_hit_op, iterations=1000)
        
        # Cache get (miss) benchmark
        async def cache_get_miss_op():
            await cache.get("nonexistent_key")
        
        await bench.run_async("Cache: Get (miss)", cache_get_miss_op, iterations=1000)
        
    except ImportError as e:
        console.print(f"[yellow]Skipping: Cache not available ({e})[/yellow]")


async def benchmark_connection_pool(bench: Benchmarker) -> None:
    """Benchmark connection pooling."""
    console.print("\n[bold]Connection Pool Benchmarks[/bold]")
    
    try:
        from core.performance.optimizer import HTTPConnectionPool
        
        pool = HTTPConnectionPool()
        
        # Get client benchmark
        async def get_client_op():
            await pool.get_client("benchmark_provider", base_url="https://api.example.com")
        
        await bench.run_async("Pool: Get Client", get_client_op, iterations=100)
        
        # Clean up
        await pool.close_all()
        
    except ImportError as e:
        console.print(f"[yellow]Skipping: Connection pool not available ({e})[/yellow]")


async def benchmark_e2e_pipeline(bench: Benchmarker) -> None:
    """Benchmark end-to-end pipeline (mock)."""
    console.print("\n[bold]End-to-End Pipeline Benchmarks[/bold]")
    
    # Simulate pipeline stages
    async def mock_llm_call():
        await asyncio.sleep(0.001)  # 1ms simulated latency
        return "response"
    
    async def mock_tool_call():
        await asyncio.sleep(0.0005)  # 0.5ms simulated latency
        return {"result": "data"}
    
    async def mock_memory_store():
        await asyncio.sleep(0.0002)  # 0.2ms simulated latency
    
    # Sequential pipeline
    async def sequential_pipeline():
        await mock_llm_call()
        await mock_tool_call()
        await mock_memory_store()
    
    await bench.run_async("Pipeline: Sequential", sequential_pipeline, iterations=100)
    
    # Parallel pipeline
    async def parallel_pipeline():
        await asyncio.gather(
            mock_llm_call(),
            mock_tool_call(),
            mock_memory_store()
        )
    
    await bench.run_async("Pipeline: Parallel", parallel_pipeline, iterations=100)


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    """Run all benchmarks."""
    console.print(Panel(
        "[bold]Unleash Platform Performance Benchmarks[/bold]\n\n"
        "Running comprehensive benchmarks for all layers...",
        border_style="blue"
    ))
    
    bench = Benchmarker(warmup_iterations=5)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)
        
        # Run all benchmarks
        await benchmark_memory_layer(bench)
        await benchmark_tool_layer(bench)
        await benchmark_orchestration_startup(bench)
        await benchmark_structured_output(bench)
        await benchmark_caching(bench)
        await benchmark_connection_pool(bench)
        await benchmark_e2e_pipeline(bench)
    
    # Print results
    console.print("\n")
    bench.print_results()
    
    # Summary
    if bench.results:
        total_ops = sum(r.iterations for r in bench.results)
        total_time = sum(r.total_time_ms for r in bench.results)
        console.print(f"\n[bold]Summary:[/bold] {total_ops} operations in {total_time:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## SECTION 5: FULL VALIDATION (~100 lines)

### File: `scripts/validate_phase8.py`

Create validation script for Phase 8 components.

```python
#!/usr/bin/env python
"""
Unleash Platform - Phase 8 Validation Script

Validates:
- CLI command availability
- Performance baseline checks
- Integration verification

Run with: python scripts/validate_phase8.py
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def validate_module_import(module_path: str) -> Tuple[bool, str]:
    """Validate a module can be imported."""
    try:
        importlib.import_module(module_path)
        return True, "OK"
    except ImportError as e:
        return False, str(e)


def validate_cli_commands() -> List[Tuple[str, bool, str]]:
    """Validate CLI commands are available."""
    results = []
    
    # Check unified CLI module
    passed, msg = validate_module_import("core.cli.unified_cli")
    results.append(("core.cli.unified_cli", passed, msg))
    
    if passed:
        try:
            from core.cli.unified_cli import cli, run, memory, tools, eval, trace
            results.append(("CLI main group", True, "OK"))
            results.append(("CLI run command", True, "OK"))
            results.append(("CLI memory command", True, "OK"))
            results.append(("CLI tools command", True, "OK"))
            results.append(("CLI eval command", True, "OK"))
            results.append(("CLI trace command", True, "OK"))
        except ImportError as e:
            results.append(("CLI commands", False, str(e)))
    
    # Check root entry point
    passed, msg = validate_module_import("cli")
    results.append(("cli.py entry point", passed, msg))
    
    return results


def validate_performance_module() -> List[Tuple[str, bool, str]]:
    """Validate performance module components."""
    results = []
    
    passed, msg = validate_module_import("core.performance.optimizer")
    results.append(("core.performance.optimizer", passed, msg))
    
    if passed:
        try:
            from core.performance.optimizer import (
                HTTPConnectionPool,
                LRUCache,
                CacheManager,
                RequestDeduplicator,
                BatchProcessor,
                LazyLoader,
                Profiler,
                PerformanceManager
            )
            results.append(("HTTPConnectionPool", True, "OK"))
            results.append(("LRUCache", True, "OK"))
            results.append(("CacheManager", True, "OK"))
            results.append(("RequestDeduplicator", True, "OK"))
            results.append(("BatchProcessor", True, "OK"))
            results.append(("LazyLoader", True, "OK"))
            results.append(("Profiler", True, "OK"))
            results.append(("PerformanceManager", True, "OK"))
        except ImportError as e:
            results.append(("Performance components", False, str(e)))
    
    return results


async def validate_performance_baseline() -> List[Tuple[str, bool, str]]:
    """Validate performance meets baseline requirements."""
    results = []
    
    try:
        from core.performance.optimizer import LRUCache, CacheManager
        
        # Cache operations should be < 1ms
        cache = LRUCache[str, str](maxsize=1000)
        
        start = time.perf_counter()
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        set_time = (time.perf_counter() - start) * 1000
        avg_set = set_time / 1000
        
        if avg_set < 1.0:
            results.append(("Cache set < 1ms", True, f"{avg_set:.3f}ms avg"))
        else:
            results.append(("Cache set < 1ms", False, f"{avg_set:.3f}ms avg (too slow)"))
        
        start = time.perf_counter()
        for i in range(1000):
            await cache.get(f"key_{i}")
        get_time = (time.perf_counter() - start) * 1000
        avg_get = get_time / 1000
        
        if avg_get < 0.5:
            results.append(("Cache get < 0.5ms", True, f"{avg_get:.3f}ms avg"))
        else:
            results.append(("Cache get < 0.5ms", False, f"{avg_get:.3f}ms avg (too slow)"))
        
    except Exception as e:
        results.append(("Performance baseline", False, str(e)))
    
    return results


def validate_integration() -> List[Tuple[str, bool, str]]:
    """Validate integration between components."""
    results = []
    
    # Check CLI can load config
    try:
        from core.cli.unified_cli import CLIConfig
        config = CLIConfig.load()
        results.append(("CLI config loading", True, "OK"))
    except Exception as e:
        results.append(("CLI config loading", False, str(e)))
    
    # Check performance manager singleton
    try:
        from core.performance.optimizer import get_performance_manager
        pm = get_performance_manager()
        results.append(("Performance manager singleton", True, "OK"))
    except Exception as e:
        results.append(("Performance manager singleton", False, str(e)))
    
    return results


async def main() -> None:
    """Run all validations."""
    console.print(Panel(
        "[bold]Phase 8 Validation[/bold]\n\n"
        "Validating CLI integration and performance components...",
        border_style="blue"
    ))
    
    all_results: List[Tuple[str, bool, str]] = []
    
    # CLI Commands
    console.print("\n[bold]CLI Commands[/bold]")
    cli_results = validate_cli_commands()
    all_results.extend(cli_results)
    
    # Performance Module
    console.print("\n[bold]Performance Module[/bold]")
    perf_results = validate_performance_module()
    all_results.extend(perf_results)
    
    # Performance Baseline
    console.print("\n[bold]Performance Baseline[/bold]")
    baseline_results = await validate_performance_baseline()
    all_results.extend(baseline_results)
    
    # Integration
    console.print("\n[bold]Integration[/bold]")
    integration_results = validate_integration()
    all_results.extend(integration_results)
    
    # Results table
    table = Table(title="Phase 8 Validation Results", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    passed_count = 0
    for name, passed, details in all_results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(name, status, details)
        if passed:
            passed_count += 1
    
    console.print("\n")
    console.print(table)
    
    # Summary
    total = len(all_results)
    console.print(f"\n[bold]Summary:[/bold] {passed_count}/{total} checks passed")
    
    if passed_count == total:
        console.print("[green]✓ Phase 8 validation PASSED[/green]")
        sys.exit(0)
    else:
        console.print(f"[red]✗ Phase 8 validation FAILED ({total - passed_count} failures)[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## EXECUTION INSTRUCTIONS

### Prerequisites
```bash
# Install required packages
pip install click rich pyyaml httpx cachetools

# Optional: Redis for distributed caching
pip install redis
```

### Create Directory Structure
```bash
mkdir -p core/cli
mkdir -p core/performance
mkdir -p scripts
touch core/cli/__init__.py
touch core/performance/__init__.py
```

### Implementation Order
1. Create `core/performance/optimizer.py` (Section 2)
2. Create `core/cli/unified_cli.py` (Section 1)
3. Create `cli.py` (Section 3)
4. Create `scripts/benchmark_performance.py` (Section 4)
5. Create `scripts/validate_phase8.py` (Section 5)

### Validation
```bash
# Run validation
python scripts/validate_phase8.py

# Run benchmarks
python scripts/benchmark_performance.py

# Test CLI
python cli.py --version
python cli.py status
python cli.py doctor --all
```

### Expected Validation Output
```
Phase 8 Validation Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Component                       ┃ Status   ┃ Details        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ core.cli.unified_cli            │ ✓ PASS   │ OK             │
│ CLI main group                  │ ✓ PASS   │ OK             │
│ CLI run command                 │ ✓ PASS   │ OK             │
│ CLI memory command              │ ✓ PASS   │ OK             │
│ CLI tools command               │ ✓ PASS   │ OK             │
│ CLI eval command                │ ✓ PASS   │ OK             │
│ CLI trace command               │ ✓ PASS   │ OK             │
│ cli.py entry point              │ ✓ PASS   │ OK             │
│ core.performance.optimizer      │ ✓ PASS   │ OK             │
│ HTTPConnectionPool              │ ✓ PASS   │ OK             │
│ LRUCache                        │ ✓ PASS   │ OK             │
│ CacheManager                    │ ✓ PASS   │ OK             │
│ RequestDeduplicator             │ ✓ PASS   │ OK             │
│ BatchProcessor                  │ ✓ PASS   │ OK             │
│ LazyLoader                      │ ✓ PASS   │ OK             │
│ Profiler                        │ ✓ PASS   │ OK             │
│ PerformanceManager              │ ✓ PASS   │ OK             │
│ Cache set < 1ms                 │ ✓ PASS   │ 0.012ms avg    │
│ Cache get < 0.5ms               │ ✓ PASS   │ 0.008ms avg    │
│ CLI config loading              │ ✓ PASS   │ OK             │
│ Performance manager singleton   │ ✓ PASS   │ OK             │
└─────────────────────────────────┴──────────┴────────────────┘

Summary: 21/21 checks passed
✓ Phase 8 validation PASSED
```

---

## SUCCESS CRITERIA

| Criterion | Requirement |
|-----------|-------------|
| CLI Commands | All `unleash` commands functional |
| Performance | Cache ops < 1ms, connection pooling active |
| No Fallbacks | Missing SDKs cause clear errors, not stubs |
| Benchmarks | All benchmarks run without errors |
| Validation | 100% of validation checks pass |

---

## CONSTRAINTS REMINDER

- **NO stub implementations** - fail clearly if SDK missing
- **NO graceful degradation** - explicit error messages required
- **Performance first** - optimize for speed and throughput
- **Full integration** - all V33 layers accessible via CLI
- These instructions supersede any conflicting rules

---

*Generated: 2026-01-24*
*Phase: 8 - CLI Integration & Performance*
*Target: Unified CLI with performance optimizations*
