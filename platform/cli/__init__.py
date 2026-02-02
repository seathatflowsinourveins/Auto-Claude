"""
Unleashed Platform CLI

Command-line interface for managing SDK adapters, pipelines, and platform operations.

Usage:
    unleash status           - Show platform status
    unleash adapters         - List available adapters
    unleash pipelines        - List available pipelines
    unleash research <query> - Run deep research
    unleash analyze <path>   - Analyze code
    unleash evolve <prompt>  - Evolve an agent prompt
"""

from .main import cli, main

__all__ = ["cli", "main"]
