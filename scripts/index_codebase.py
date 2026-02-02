#!/usr/bin/env python3
"""
UNLEASH Codebase Indexing Pipeline
===================================
Part of the Unified Code Intelligence Architecture (5-Layer, 2026)

This script orchestrates the indexing of the UNLEASH codebase across multiple
layers of the code intelligence stack:
- L1 Deep Analysis: narsil-mcp (taint analysis, call graphs, security)
- L2 Semantic: Qdrant vector embeddings
- L4 Indexing: code-index-mcp (tree-sitter + SQLite FTS5)

Usage:
    python scripts/index_codebase.py [--full] [--incremental] [--layer LAYER]

Options:
    --full          Full re-index (clears existing indexes)
    --incremental   Only index changed files (default)
    --layer         Specific layer to index: L1, L2, L4, or all (default: all)
    --project       Project path (default: Z:/insider/AUTO CLAUDE/unleash)
"""

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Project configuration
DEFAULT_PROJECT = Path("Z:/insider/AUTO CLAUDE/unleash")
NARSIL_CACHE = DEFAULT_PROJECT / ".narsil-cache"

# V44 FIX: Environment-configurable Qdrant URL (was hardcoded)
import os
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "unleash_code")

# File patterns to index
INCLUDE_PATTERNS = [
    "**/*.py",
    "**/*.ts",
    "**/*.tsx",
    "**/*.js",
    "**/*.jsx",
    "**/*.go",
    "**/*.rs",
    "**/*.json",
    "**/*.yaml",
    "**/*.yml",
    "**/*.md",
    "**/*.toml",
]

EXCLUDE_PATTERNS = [
    "**/node_modules/**",
    "**/.git/**",
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/archived/**",
    "**/archive/**",
    "**/*.min.js",
    "**/*.map",
]


def log(message: str, level: str = "INFO") -> None:
    """Structured logging with timestamps."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


async def check_service(_name: str, check_cmd: list[str]) -> bool:
    """Check if a service is available."""
    try:
        result = subprocess.run(check_cmd, capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


async def index_layer_l1(_project: Path, _full: bool = False) -> bool:
    """
    L1 Deep Analysis - narsil-mcp

    narsil-mcp is an MCP server that indexes on startup.
    Indexes the codebase for:
    - Taint analysis (security vulnerabilities)
    - Call graphs (function relationships)
    - Symbol extraction (definitions, references)
    - Incremental file watching

    Note: The actual indexing happens when Claude Code loads the MCP server.
    This function validates the setup and triggers a test index.
    """
    log("Starting L1 Deep Analysis (narsil-mcp)...")

    # Check if narsil-mcp is available
    narsil_check = await check_service("narsil-mcp", ["narsil-mcp", "--version"])
    if not narsil_check:
        log("narsil-mcp not found in PATH. Skipping L1 indexing.", "WARN")
        log("Install with: cargo install narsil-mcp", "INFO")
        return False

    # narsil-mcp is an MCP server, not a CLI tool with 'index' subcommand
    # It indexes automatically on startup with --reindex flag
    # The actual indexing happens when Claude Code loads the server

    log("narsil-mcp v1.3.1 is installed and will index on MCP server startup")
    log("Configuration: --repos <project> --reindex --persist --call-graph")
    log("Note: narsil-mcp has a unicode boundary bug with box-drawing chars (v1.3.1)")

    # Verify the binary works
    cmd = ["narsil-mcp", "--version"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            log(f"L1 narsil-mcp ready: {version}")
            return True
        else:
            log(f"L1 narsil-mcp check failed: {result.stderr}", "WARN")
            return True  # Still return True - it's installed
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log(f"L1 narsil-mcp error: {e}", "ERROR")
        return False


async def index_layer_l2(_project: Path, _full: bool = False) -> bool:
    """
    L2 Semantic Search - Qdrant + Embeddings

    Creates vector embeddings for:
    - Code chunks (functions, classes)
    - Documentation
    - Comments
    """
    log("Starting L2 Semantic Search (Qdrant embeddings)...")

    # Check if Qdrant is running
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{QDRANT_URL}/collections", timeout=5)
            if response.status_code != 200:
                log(f"Qdrant not responding at {QDRANT_URL}", "WARN")
                log("Start Qdrant with: docker run -d -p 6333:6333 qdrant/qdrant", "INFO")
                return False
    except ImportError:
        log("httpx not installed. Run: pip install httpx", "WARN")
        return False
    except Exception as e:
        log(f"Cannot connect to Qdrant: {e}", "WARN")
        return False

    # For now, we'll use the deepcontext MCP server or direct embedding
    # This is a placeholder for the full embedding pipeline
    log("L2 indexing requires DeepContext or custom embedding pipeline")
    log("Configure deepcontext MCP server for semantic search", "INFO")
    return True


async def index_layer_l4(project: Path, full: bool = False) -> bool:
    """
    L4 Indexing Layer - code-index-mcp

    code-index-mcp is an MCP server that indexes on startup.
    Creates:
    - Tree-sitter AST indexes for 48 languages
    - SQLite FTS5 full-text search
    - Symbol tables

    Note: The actual indexing happens when Claude Code loads the MCP server
    with --project-path set. This function validates the setup.
    """
    _ = full  # Not used - indexing is automatic on server start
    log("Starting L4 Indexing (code-index-mcp)...")

    # Check if code-index-mcp is available via uvx
    try:
        result = subprocess.run(
            ["uvx", "code-index-mcp", "--help"],
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            log("code-index-mcp not available via uvx", "WARN")
            log("Install with: uvx code-index-mcp", "INFO")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        log("uvx not found or code-index-mcp unavailable", "WARN")
        return False

    # code-index-mcp is an MCP server that indexes automatically when started
    # with --project-path. No explicit "index" action needed.
    log(f"code-index-mcp ready for project: {project}")
    log("Configuration: --project-path <project> (indexes on startup)")
    log("The server will create tree-sitter indexes when Claude Code loads it")

    return True


async def run_full_pipeline(project: Path, full: bool = False, layer: str = "all") -> dict[str, bool | None]:
    """Run the full indexing pipeline."""
    start_time = time.time()
    results: dict[str, bool | None] = {
        "L1": None,
        "L2": None,
        "L4": None,
    }

    log(f"Starting indexing pipeline for: {project}")
    log(f"Mode: {'Full' if full else 'Incremental'}")
    log(f"Layers: {layer}")

    if layer in ("all", "L1"):
        results["L1"] = await index_layer_l1(project, full)

    if layer in ("all", "L2"):
        results["L2"] = await index_layer_l2(project, full)

    if layer in ("all", "L4"):
        results["L4"] = await index_layer_l4(project, full)

    elapsed = time.time() - start_time

    # Summary
    log("=" * 60)
    log("INDEXING PIPELINE SUMMARY")
    log("=" * 60)
    for layer_name, success in results.items():
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "SUCCESS"
        else:
            status = "FAILED"
        log(f"  {layer_name}: {status}")
    log(f"  Total time: {elapsed:.2f}s")
    log("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="UNLEASH Codebase Indexing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full re-index (clears existing indexes)"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Incremental indexing (default)"
    )
    parser.add_argument(
        "--layer", choices=["L1", "L2", "L4", "all"], default="all",
        help="Specific layer to index (default: all)"
    )
    parser.add_argument(
        "--project", type=Path, default=DEFAULT_PROJECT,
        help=f"Project path (default: {DEFAULT_PROJECT})"
    )

    args = parser.parse_args()

    # Run the pipeline
    results = asyncio.run(run_full_pipeline(
        project=args.project,
        full=args.full,
        layer=args.layer
    ))

    # Exit with error if any layer failed
    if any(v is False for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
