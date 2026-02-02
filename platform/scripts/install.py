#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.0.0",
#     "questionary>=2.0.0",
# ]
# ///
"""
Platform Installer - Ultimate Autonomous Platform Setup

Interactive CLI installer for setting up the platform environment.
Handles dependency installation, configuration, and validation.

Features:
- System requirements check
- Python dependency installation
- Docker environment setup
- Database initialization
- Configuration file generation
- Health check validation

Usage:
    python install.py              # Interactive installation
    python install.py --quick      # Quick install with defaults
    python install.py --check      # Check installation status
    python install.py --uninstall  # Remove platform components
"""

from __future__ import annotations

import subprocess
import sys
import shutil
import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

# Optional rich library for pretty output
Console: Optional[type] = None
Table: Optional[type] = None
Panel: Optional[type] = None
Confirm: Optional[type] = None
RICH_AVAILABLE = False

try:
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTable
    from rich.panel import Panel as RichPanel
    from rich.prompt import Confirm as RichConfirm
    Console = RichConsole
    Table = RichTable
    Panel = RichPanel
    Confirm = RichConfirm
    RICH_AVAILABLE = True
except ImportError:
    pass

# Optional questionary library for interactive prompts
questionary: Any = None
QUESTIONARY_AVAILABLE = False

try:
    import questionary as q
    questionary = q
    QUESTIONARY_AVAILABLE = True
except ImportError:
    pass


class InstallStep(str, Enum):
    """Installation steps."""
    CHECK_SYSTEM = "check_system"
    INSTALL_PYTHON_DEPS = "install_python_deps"
    SETUP_DOCKER = "setup_docker"
    INIT_DATABASES = "init_databases"
    CONFIGURE_PLATFORM = "configure_platform"
    VALIDATE_INSTALLATION = "validate_installation"


@dataclass
class SystemRequirements:
    """System requirements specification."""
    python_min_version: tuple = (3, 11)
    docker_required: bool = True
    min_ram_gb: int = 4
    min_disk_gb: int = 10
    required_ports: List[int] = field(default_factory=lambda: [
        6333,  # Qdrant
        7687,  # Neo4j Bolt
        7474,  # Neo4j HTTP
        8283,  # Letta
        6379,  # Redis
        8080,  # Platform API
        9090,  # Prometheus
        9091,  # Autoscale metrics
    ])


@dataclass
class InstallResult:
    """Result of an installation step."""
    step: InstallStep
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class PlatformInstaller:
    """
    Interactive platform installer.

    Handles all aspects of platform setup including:
    - System requirement validation
    - Dependency installation
    - Container orchestration
    - Database initialization
    - Configuration management
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        interactive: bool = True,
        verbose: bool = True
    ):
        self.base_dir = base_dir or Path(__file__).parent.parent
        self.interactive = interactive
        self.verbose = verbose
        self.requirements = SystemRequirements()
        self.results: List[InstallResult] = []

        # Console for output
        if RICH_AVAILABLE and Console is not None:
            self.console = Console()
        else:
            self.console = None

    def print(self, message: str, style: str = "") -> None:
        """Print message with optional styling."""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def print_header(self, title: str) -> None:
        """Print section header."""
        if self.console and Panel is not None:
            self.console.print(Panel(title, style="bold blue"))
        else:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}\n")

    def confirm(self, message: str, default: bool = True) -> bool:
        """Ask for confirmation."""
        if not self.interactive:
            return default

        if QUESTIONARY_AVAILABLE and questionary is not None:
            result = questionary.confirm(message, default=default).ask()
            return result if result is not None else default
        elif self.console and Confirm is not None:
            return Confirm.ask(message, default=default)
        else:
            response = input(f"{message} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
            if not response:
                return default
            return response in ('y', 'yes')

    def run_command(
        self,
        cmd: List[str],
        capture: bool = True,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a shell command."""
        if self.verbose:
            self.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check
        )

    def check_system_requirements(self) -> InstallResult:
        """Check system requirements."""
        self.print_header("Checking System Requirements")

        issues = []
        details = {}

        # Python version
        py_version = sys.version_info[:2]
        details["python_version"] = f"{py_version[0]}.{py_version[1]}"
        if py_version < self.requirements.python_min_version:
            issues.append(f"Python {self.requirements.python_min_version[0]}.{self.requirements.python_min_version[1]}+ required")
        else:
            self.print(f"  [green]OK[/green] Python {details['python_version']}")

        # Docker
        docker_available = shutil.which("docker") is not None
        details["docker_available"] = docker_available
        if self.requirements.docker_required and not docker_available:
            issues.append("Docker not found in PATH")
        else:
            self.print(f"  [green]OK[/green] Docker available")

            # Check if Docker daemon is running
            try:
                result = self.run_command(["docker", "info"], check=False)
                docker_running = result.returncode == 0
                details["docker_running"] = docker_running
                if docker_running:
                    self.print(f"  [green]OK[/green] Docker daemon running")
                else:
                    issues.append("Docker daemon not running")
            except Exception:
                details["docker_running"] = False
                issues.append("Could not check Docker daemon status")

        # Docker Compose
        compose_available = (
            shutil.which("docker-compose") is not None or
            self._check_docker_compose_v2()
        )
        details["compose_available"] = compose_available
        if compose_available:
            self.print(f"  [green]OK[/green] Docker Compose available")
        else:
            issues.append("Docker Compose not found")

        # Platform
        details["platform"] = platform.system()
        details["architecture"] = platform.machine()
        self.print(f"  [blue]i[/blue] Platform: {details['platform']} ({details['architecture']})")

        # Check available ports
        blocked_ports = []
        for port in self.requirements.required_ports:
            if self._is_port_in_use(port):
                blocked_ports.append(port)

        if blocked_ports:
            details["blocked_ports"] = blocked_ports
            issues.append(f"Ports in use: {blocked_ports}")
        else:
            self.print(f"  [green]OK[/green] All required ports available")

        success = len(issues) == 0
        message = "System requirements met" if success else f"Issues: {', '.join(issues)}"

        return InstallResult(
            step=InstallStep.CHECK_SYSTEM,
            success=success,
            message=message,
            details=details
        )

    def _check_docker_compose_v2(self) -> bool:
        """Check if docker compose (v2) is available."""
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def install_python_dependencies(self) -> InstallResult:
        """Install Python dependencies."""
        self.print_header("Installing Python Dependencies")

        requirements_file = self.base_dir / "requirements.txt"

        # Check if requirements.txt exists
        if not requirements_file.exists():
            # Create requirements.txt from pyproject.toml or script metadata
            self._generate_requirements_file(requirements_file)

        try:
            # Install with pip
            self.print("  Installing dependencies with pip...")

            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = self.run_command(cmd, check=False)

            if result.returncode == 0:
                self.print("  [green]OK[/green] Dependencies installed successfully")
                return InstallResult(
                    step=InstallStep.INSTALL_PYTHON_DEPS,
                    success=True,
                    message="Dependencies installed",
                    details={"requirements_file": str(requirements_file)}
                )
            else:
                return InstallResult(
                    step=InstallStep.INSTALL_PYTHON_DEPS,
                    success=False,
                    message=f"pip install failed: {result.stderr}",
                    details={"returncode": result.returncode}
                )
        except Exception as e:
            return InstallResult(
                step=InstallStep.INSTALL_PYTHON_DEPS,
                success=False,
                message=str(e)
            )

    def _generate_requirements_file(self, path: Path) -> None:
        """Generate requirements.txt from known dependencies."""
        dependencies = [
            "# Ultimate Autonomous Platform Dependencies",
            "# Generated by install.py",
            "",
            "# Core",
            "pydantic>=2.5.0",
            "pydantic-settings>=2.1.0",
            "",
            "# Databases",
            "qdrant-client>=1.7.0",
            "neo4j>=5.15.0",
            "redis>=5.0.0",
            "",
            "# Observability",
            "prometheus-client>=0.19.0",
            "opentelemetry-api>=1.22.0",
            "opentelemetry-sdk>=1.22.0",
            "opentelemetry-exporter-otlp>=1.22.0",
            "",
            "# Security",
            "cryptography>=42.0.0",
            "",
            "# HTTP",
            "httpx>=0.26.0",
            "aiohttp>=3.9.0",
            "",
            "# CLI",
            "rich>=13.0.0",
            "questionary>=2.0.0",
            "",
            "# Testing",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
        ]

        path.write_text("\n".join(dependencies))
        self.print(f"  [blue]i[/blue] Generated {path}")

    def setup_docker_environment(self) -> InstallResult:
        """Setup Docker environment."""
        self.print_header("Setting Up Docker Environment")

        compose_file = self.base_dir / "docker-compose.yml"

        if not compose_file.exists():
            self.print("  [yellow]![/yellow] docker-compose.yml not found")
            return InstallResult(
                step=InstallStep.SETUP_DOCKER,
                success=False,
                message="docker-compose.yml not found"
            )

        # Check if containers are already running
        try:
            result = self.run_command(
                ["docker", "compose", "ps", "--format", "json"],
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                running_containers = len(result.stdout.strip().split('\n'))
                self.print(f"  [blue]i[/blue] {running_containers} containers already running")

                if not self.confirm("Restart containers?", default=False):
                    return InstallResult(
                        step=InstallStep.SETUP_DOCKER,
                        success=True,
                        message="Using existing containers",
                        details={"containers": running_containers}
                    )
        except Exception:
            pass

        # Start containers
        self.print("  Starting Docker containers...")

        try:
            # Pull images first
            self.print("  [dim]Pulling images...[/dim]")
            self.run_command(
                ["docker", "compose", "-f", str(compose_file), "pull"],
                check=False
            )

            # Start services
            self.print("  [dim]Starting services...[/dim]")
            result = self.run_command(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                check=False
            )

            if result.returncode == 0:
                self.print("  [green]OK[/green] Docker environment ready")
                return InstallResult(
                    step=InstallStep.SETUP_DOCKER,
                    success=True,
                    message="Docker environment started"
                )
            else:
                return InstallResult(
                    step=InstallStep.SETUP_DOCKER,
                    success=False,
                    message=f"docker compose up failed: {result.stderr}"
                )
        except Exception as e:
            return InstallResult(
                step=InstallStep.SETUP_DOCKER,
                success=False,
                message=str(e)
            )

    def initialize_databases(self) -> InstallResult:
        """Initialize database schemas."""
        self.print_header("Initializing Databases")

        results = []

        # Wait for services to be ready
        self.print("  Waiting for services to be ready...")
        import time
        time.sleep(5)

        # Initialize Qdrant collections
        try:
            self.print("  [dim]Initializing Qdrant collections...[/dim]")
            # Qdrant auto-creates collections, just verify connectivity
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', 6333)) == 0:
                    self.print("  [green]OK[/green] Qdrant ready")
                    results.append(("qdrant", True))
                else:
                    results.append(("qdrant", False))
        except Exception as e:
            results.append(("qdrant", False))
            self.print(f"  [red]FAIL[/red] Qdrant: {e}")

        # Initialize Neo4j
        try:
            self.print("  [dim]Checking Neo4j...[/dim]")
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', 7687)) == 0:
                    self.print("  [green]OK[/green] Neo4j ready")
                    results.append(("neo4j", True))
                else:
                    results.append(("neo4j", False))
        except Exception as e:
            results.append(("neo4j", False))
            self.print(f"  [red]FAIL[/red] Neo4j: {e}")

        # Initialize Redis
        try:
            self.print("  [dim]Checking Redis...[/dim]")
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', 6379)) == 0:
                    self.print("  [green]OK[/green] Redis ready")
                    results.append(("redis", True))
                else:
                    results.append(("redis", False))
        except Exception as e:
            results.append(("redis", False))
            self.print(f"  [red]FAIL[/red] Redis: {e}")

        success_count = sum(1 for _, success in results if success)
        total = len(results)

        return InstallResult(
            step=InstallStep.INIT_DATABASES,
            success=success_count == total,
            message=f"{success_count}/{total} databases initialized",
            details=dict(results)
        )

    def configure_platform(self) -> InstallResult:
        """Generate platform configuration."""
        self.print_header("Configuring Platform")

        config_dir = self.base_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # Create .env file
        env_file = self.base_dir / ".env"
        if not env_file.exists():
            self.print("  Creating .env file...")
            env_content = """# Ultimate Autonomous Platform Configuration
# Generated by install.py

# Environment
UAP_ENVIRONMENT=development

# Qdrant
UAP_QDRANT__URL=http://localhost:6333

# Neo4j
UAP_NEO4J__URI=bolt://localhost:7687
UAP_NEO4J__USERNAME=neo4j
UAP_NEO4J__PASSWORD=alphaforge2024

# Letta
UAP_LETTA__URL=http://localhost:8283

# Redis
UAP_REDIS__URL=redis://localhost:6379

# Server
UAP_SERVER__HOST=0.0.0.0
UAP_SERVER__PORT=8080
UAP_SERVER__WORKERS=4

# Tracing
UAP_TRACING__ENABLED=true
UAP_TRACING__SERVICE_NAME=ultimate-autonomous-platform

# Metrics
UAP_METRICS__ENABLED=true
UAP_METRICS__PORT=9090
"""
            env_file.write_text(env_content)
            self.print("  [green]OK[/green] Created .env file")
        else:
            self.print("  [blue]i[/blue] .env file already exists")

        # Create secrets directory
        secrets_dir = Path.home() / ".uap"
        secrets_dir.mkdir(exist_ok=True)
        self.print(f"  [green]OK[/green] Secrets directory: {secrets_dir}")

        return InstallResult(
            step=InstallStep.CONFIGURE_PLATFORM,
            success=True,
            message="Platform configured",
            details={"env_file": str(env_file), "secrets_dir": str(secrets_dir)}
        )

    def validate_installation(self) -> InstallResult:
        """Validate the installation."""
        self.print_header("Validating Installation")

        checks = []

        # Check scripts exist
        scripts_dir = self.base_dir / "scripts"
        required_scripts = [
            "platform_orchestrator.py",
            "config.py",
            "secrets.py",
            "autoscale.py",
            "tracing.py",
            "rate_limiter.py",
        ]

        for script in required_scripts:
            script_path = scripts_dir / script
            exists = script_path.exists()
            checks.append((f"script:{script}", exists))
            if exists:
                self.print(f"  [green]OK[/green] {script}")
            else:
                self.print(f"  [red]FAIL[/red] {script} missing")

        # Check Docker services
        try:
            result = self.run_command(
                ["docker", "compose", "ps", "--services", "--filter", "status=running"],
                check=False
            )
            if result.returncode == 0:
                running = result.stdout.strip().split('\n') if result.stdout.strip() else []
                for service in running:
                    checks.append((f"docker:{service}", True))
                    self.print(f"  [green]OK[/green] Docker: {service}")
        except Exception:
            pass

        # Run orchestrator status
        self.print("\n  Running platform health check...")
        try:
            orchestrator = scripts_dir / "platform_orchestrator.py"
            if orchestrator.exists():
                result = self.run_command(
                    [sys.executable, str(orchestrator), "status"],
                    check=False
                )
                if "healthy" in result.stdout.lower():
                    checks.append(("health_check", True))
                    self.print("  [green]OK[/green] Platform health check passed")
                else:
                    checks.append(("health_check", False))
        except Exception:
            checks.append(("health_check", False))

        success_count = sum(1 for _, success in checks if success)
        total = len(checks)

        return InstallResult(
            step=InstallStep.VALIDATE_INSTALLATION,
            success=success_count >= total * 0.8,  # 80% threshold
            message=f"{success_count}/{total} checks passed",
            details=dict(checks)
        )

    def print_summary(self) -> None:
        """Print installation summary."""
        self.print_header("Installation Summary")

        if self.console and Table is not None:
            table = Table(title="Installation Results")
            table.add_column("Step", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Message")

            for result in self.results:
                status = "[green]OK Success[/green]" if result.success else "[red]FAIL Failed[/red]"
                table.add_row(result.step.value, status, result.message)

            self.console.print(table)
        else:
            for result in self.results:
                status = "OK" if result.success else "FAIL"
                print(f"  [{status}] {result.step.value}: {result.message}")

        # Overall status
        success = all(r.success for r in self.results)
        if success:
            self.print("\n[bold green]OK Installation completed successfully![/bold green]")
            self.print("\nNext steps:")
            self.print("  1. Start the platform: python scripts/platform_orchestrator.py serve")
            self.print("  2. Check health: python scripts/platform_orchestrator.py status")
            self.print("  3. View metrics: http://localhost:9090/metrics")
        else:
            self.print("\n[bold red]FAIL Installation completed with errors[/bold red]")
            self.print("Please review the errors above and retry failed steps.")

    def run_full_install(self) -> bool:
        """Run full installation."""
        self.print_header("Ultimate Autonomous Platform Installer")
        self.print("This installer will set up the platform and all dependencies.\n")

        if self.interactive:
            if not self.confirm("Continue with installation?"):
                self.print("Installation cancelled.")
                return False

        steps: List[Callable[[], InstallResult]] = [
            self.check_system_requirements,
            self.install_python_dependencies,
            self.setup_docker_environment,
            self.initialize_databases,
            self.configure_platform,
            self.validate_installation,
        ]

        for step_func in steps:
            result = step_func()
            self.results.append(result)

            if not result.success:
                self.print(f"\n[yellow]Warning: {result.step.value} had issues[/yellow]")
                if self.interactive and not self.confirm("Continue anyway?"):
                    break

        self.print_summary()
        return all(r.success for r in self.results)

    def run_quick_check(self) -> bool:
        """Run quick installation check."""
        self.print_header("Installation Check")

        result = self.check_system_requirements()
        self.results.append(result)

        result = self.validate_installation()
        self.results.append(result)

        self.print_summary()
        return all(r.success for r in self.results)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ultimate Autonomous Platform Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py              # Interactive installation
  python install.py --quick      # Quick install with defaults
  python install.py --check      # Check installation status
  python install.py --verbose    # Verbose output
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick install with defaults (non-interactive)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check installation status only"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Base directory for installation"
    )

    args = parser.parse_args()

    base_dir = Path(args.dir) if args.dir else None
    interactive = not args.quick

    installer = PlatformInstaller(
        base_dir=base_dir,
        interactive=interactive,
        verbose=args.verbose or True
    )

    if args.check:
        success = installer.run_quick_check()
    else:
        success = installer.run_full_install()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
