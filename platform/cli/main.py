"""
Unleashed Platform CLI - Main Entry Point

Provides interactive command-line interface for SDK operations.
Supports both command mode and interactive REPL mode.
"""

import asyncio
import sys
import json
from typing import Optional, List
from pathlib import Path
import argparse

# ASCII Art Banner
BANNER = """
 _   _       _                 _              _
| | | |_ __ | | ___  __ _ ___| |__   ___  __| |
| | | | '_ \\| |/ _ \\/ _` / __| '_ \\ / _ \\/ _` |
| |_| | | | | |  __/ (_| \\__ \\ | | |  __/ (_| |
 \\___/|_| |_|_|\\___|\\__,_|___/_| |_|\\___|\\__,_|

          SDK Stack Orchestrator v2.0
"""


def print_banner():
    """Print the CLI banner."""
    print(BANNER)


def get_adapter_status():
    """Get status of all adapters."""
    try:
        from ..adapters import get_adapter_status
        return get_adapter_status()
    except ImportError:
        return {}


def get_pipeline_status():
    """Get status of all pipelines."""
    try:
        from ..pipelines import get_pipeline_status
        return get_pipeline_status()
    except ImportError:
        return {}


def get_core_status():
    """Get status of core components."""
    status = {}

    # Check async executor
    try:
        from ..core.async_executor import AsyncExecutor
        status["async_executor"] = True
    except ImportError:
        status["async_executor"] = False

    # Check parallel orchestrator
    try:
        from ..core.parallel_orchestrator import ParallelOrchestrator
        status["parallel_orchestrator"] = True
    except ImportError:
        status["parallel_orchestrator"] = False

    # Check caching
    try:
        from ..core.caching import MemoryCache
        status["caching"] = True
    except ImportError:
        status["caching"] = False

    # Check error handling
    try:
        from ..core.error_handling import ErrorHandler
        status["error_handling"] = True
    except ImportError:
        status["error_handling"] = False

    return status


def cmd_status(args):
    """Show platform status."""
    print_banner()
    print("\n" + "=" * 60)
    print("PLATFORM STATUS")
    print("=" * 60)

    # Adapters
    print("\n[ADAPTERS]")
    adapters = get_adapter_status()
    if adapters:
        for name, info in adapters.items():
            status = "[OK]" if info.get("available") else "[--]"
            version = info.get("version", "N/A")
            print(f"  {status} {name:20} v{version}")
    else:
        print("  No adapters registered")

    # Pipelines
    print("\n[PIPELINES]")
    pipelines = get_pipeline_status()
    if pipelines:
        for name, info in pipelines.items():
            status = "[OK]" if info.get("available") else "[--]"
            deps = ", ".join(info.get("dependencies", []))
            print(f"  {status} {name:25} deps: {deps or 'none'}")
    else:
        print("  No pipelines registered")

    # Core Components
    print("\n[CORE COMPONENTS]")
    core = get_core_status()
    for component, available in core.items():
        status = "[OK]" if available else "[--]"
        print(f"  {status} {component}")

    print("\n" + "=" * 60)


def cmd_adapters(args):
    """List adapters with details."""
    print("\n" + "=" * 50)
    print("AVAILABLE ADAPTERS")
    print("=" * 50)

    adapters = get_adapter_status()

    if not adapters:
        print("\nNo adapters registered.")
        print("\nTo register adapters, import them:")
        print("  from platform.adapters import dspy_adapter")
        return

    for name, info in adapters.items():
        available = info.get("available", False)
        version = info.get("version", "N/A")
        initialized = info.get("initialized", False)

        print(f"\n{name}")
        print(f"  Available:   {'Yes' if available else 'No'}")
        print(f"  Version:     {version}")
        print(f"  Initialized: {'Yes' if initialized else 'No'}")

        # Show adapter-specific info
        if name == "dspy":
            print("  Purpose:     Declarative prompt programming")
            print("  Install:     pip install dspy-ai")
        elif name == "langgraph":
            print("  Purpose:     State graph workflows")
            print("  Install:     pip install langgraph")
        elif name == "mem0":
            print("  Purpose:     Unified memory layer")
            print("  Install:     pip install mem0ai")
        elif name == "textgrad":
            print("  Purpose:     Gradient-based prompt optimization")
            print("  Install:     pip install textgrad")
        elif name == "aider":
            print("  Purpose:     AI pair programming")
            print("  Install:     pip install aider-chat")


def cmd_pipelines(args):
    """List pipelines with details."""
    print("\n" + "=" * 50)
    print("AVAILABLE PIPELINES")
    print("=" * 50)

    pipelines = get_pipeline_status()

    if not pipelines:
        print("\nNo pipelines registered.")
        return

    for name, info in pipelines.items():
        available = info.get("available", False)
        deps = info.get("dependencies", [])

        print(f"\n{name}")
        print(f"  Available:    {'Yes' if available else 'No'}")
        print(f"  Dependencies: {', '.join(deps) if deps else 'none'}")

        # Show pipeline-specific info
        if name == "deep_research":
            print("  Purpose:      Multi-source research with synthesis")
        elif name == "self_improvement":
            print("  Purpose:      Genetic + gradient prompt evolution")
        elif name == "code_analysis":
            print("  Purpose:      AST analysis and code quality")
        elif name == "agent_evolution":
            print("  Purpose:      Population-based agent optimization")


async def cmd_research(args):
    """Run deep research pipeline."""
    query = " ".join(args.query)
    depth = args.depth or "standard"

    print(f"\nResearching: {query}")
    print(f"Depth: {depth}")
    print("-" * 50)

    try:
        from ..pipelines import get_deep_research_pipeline
        pipeline_cls = get_deep_research_pipeline()

        if not pipeline_cls:
            print("Error: Deep research pipeline not available")
            print("Install dependencies: pip install exa-py firecrawl-py")
            return

        pipeline = pipeline_cls()
        result = await pipeline.research(query, depth=depth)

        print("\n[RESULTS]")
        print(f"Sources found: {len(result.sources) if hasattr(result, 'sources') else 'N/A'}")
        print(f"\n{result.summary if hasattr(result, 'summary') else str(result)}")

    except Exception as e:
        print(f"Error: {e}")


async def cmd_analyze(args):
    """Run code analysis pipeline."""
    path = args.path
    depth = args.depth or "standard"

    print(f"\nAnalyzing: {path}")
    print(f"Depth: {depth}")
    print("-" * 50)

    try:
        from ..pipelines import get_code_analysis_pipeline
        pipeline_cls = get_code_analysis_pipeline()

        if not pipeline_cls:
            print("Error: Code analysis pipeline not available")
            return

        pipeline = pipeline_cls()
        result = await pipeline.analyze(path, depth=depth)

        print("\n[ANALYSIS RESULTS]")
        print(f"Files analyzed: {result.files_analyzed if hasattr(result, 'files_analyzed') else 'N/A'}")
        print(f"Total LOC: {result.total_loc if hasattr(result, 'total_loc') else 'N/A'}")
        print(f"Health score: {result.health_score if hasattr(result, 'health_score') else 'N/A'}")

        if hasattr(result, 'issues') and result.issues:
            print(f"\nIssues found: {len(result.issues)}")
            for issue in result.issues[:10]:
                print(f"  - {issue}")

    except Exception as e:
        print(f"Error: {e}")


async def cmd_evolve(args):
    """Run agent evolution pipeline."""
    prompt = " ".join(args.prompt)
    generations = args.generations or 10

    print(f"\nEvolving prompt:")
    print(f"  {prompt[:100]}...")
    print(f"Generations: {generations}")
    print("-" * 50)

    try:
        from ..pipelines import get_agent_evolution_pipeline
        pipeline_cls = get_agent_evolution_pipeline()

        if not pipeline_cls:
            print("Error: Agent evolution pipeline not available")
            return

        pipeline = pipeline_cls(max_generations=generations)

        # Simple evaluation function
        async def eval_fn(genome, task):
            return 0.5  # Placeholder

        result = await pipeline.evolve(
            base_prompt=prompt,
            task_prompts={},
            tasks=[{"name": "test", "description": "Test task"}],
            evaluation_fn=eval_fn,
        )

        print("\n[EVOLUTION RESULTS]")
        print(f"Generations completed: {result.generations_completed}")
        print(f"Final fitness: {result.best_genome.fitness:.4f}")
        print(f"\nBest prompt:")
        print(f"  {result.best_genome.system_prompt[:200]}...")

    except Exception as e:
        print(f"Error: {e}")


def cmd_config(args):
    """Show or set configuration."""
    if args.key and args.value:
        print(f"Setting {args.key} = {args.value}")
        # TODO: Implement config persistence
    else:
        print("\n[CONFIGURATION]")
        print("  No configuration set")
        print("\nAvailable settings:")
        print("  default_model    - Default LLM model")
        print("  cache_dir        - Cache directory path")
        print("  log_level        - Logging level (debug/info/warning/error)")


def cmd_interactive(args):
    """Start interactive REPL mode."""
    print_banner()
    print("Interactive mode. Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            user_input = input("unleash> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "help":
                print("\nCommands:")
                print("  status     - Show platform status")
                print("  adapters   - List adapters")
                print("  pipelines  - List pipelines")
                print("  research   - Start research (usage: research <query>)")
                print("  analyze    - Analyze code (usage: analyze <path>)")
                print("  evolve     - Evolve prompt (usage: evolve <prompt>)")
                print("  exit       - Exit interactive mode")
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "status":
                cmd_status(None)
            elif cmd == "adapters":
                cmd_adapters(None)
            elif cmd == "pipelines":
                cmd_pipelines(None)
            elif cmd == "research" and len(parts) > 1:
                class Args:
                    query = parts[1:]
                    depth = "standard"
                asyncio.run(cmd_research(Args()))
            elif cmd == "analyze" and len(parts) > 1:
                class Args:
                    path = parts[1]
                    depth = "standard"
                asyncio.run(cmd_analyze(Args()))
            elif cmd == "evolve" and len(parts) > 1:
                class Args:
                    prompt = parts[1:]
                    generations = 10
                asyncio.run(cmd_evolve(Args()))
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {e}")


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="unleash",
        description="Unleashed Platform CLI - SDK Stack Orchestrator",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    subparsers.add_parser("status", help="Show platform status")

    # Adapters command
    subparsers.add_parser("adapters", help="List available adapters")

    # Pipelines command
    subparsers.add_parser("pipelines", help="List available pipelines")

    # Research command
    research_parser = subparsers.add_parser("research", help="Run deep research")
    research_parser.add_argument("query", nargs="+", help="Research query")
    research_parser.add_argument("--depth", choices=["quick", "standard", "deep", "comprehensive"],
                                 default="standard", help="Research depth")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code")
    analyze_parser.add_argument("path", help="Path to analyze")
    analyze_parser.add_argument("--depth", choices=["quick", "standard", "deep"],
                               default="standard", help="Analysis depth")

    # Evolve command
    evolve_parser = subparsers.add_parser("evolve", help="Evolve agent prompt")
    evolve_parser.add_argument("prompt", nargs="+", help="Initial prompt")
    evolve_parser.add_argument("--generations", type=int, default=10,
                              help="Number of evolution generations")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show/set configuration")
    config_parser.add_argument("key", nargs="?", help="Configuration key")
    config_parser.add_argument("value", nargs="?", help="Configuration value")

    # Interactive command
    subparsers.add_parser("interactive", aliases=["i", "repl"],
                          help="Start interactive mode")

    return parser


def cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        # Default to status
        cmd_status(args)
        return

    if args.command == "status":
        cmd_status(args)
    elif args.command == "adapters":
        cmd_adapters(args)
    elif args.command == "pipelines":
        cmd_pipelines(args)
    elif args.command == "research":
        asyncio.run(cmd_research(args))
    elif args.command == "analyze":
        asyncio.run(cmd_analyze(args))
    elif args.command == "evolve":
        asyncio.run(cmd_evolve(args))
    elif args.command == "config":
        cmd_config(args)
    elif args.command in ("interactive", "i", "repl"):
        cmd_interactive(args)
    else:
        parser.print_help()


def main():
    """Alternative entry point."""
    cli()


if __name__ == "__main__":
    cli()
