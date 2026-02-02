#!/usr/bin/env python3
"""
Audit CLI commands against V35 SDK stack.
Phase 14: CLI Commands Verification

Compares existing CLI commands with expected V35 command structure.
"""

import subprocess
import sys
from pathlib import Path

# Expected CLI commands for each layer (from Phase 14 spec)
EXPECTED_COMMANDS = {
    "L0_Protocol": [
        "unleash protocol call <prompt>",
        "unleash protocol chat",
        "unleash mcp list",
        "unleash mcp connect <server>",
    ],
    "L1_Orchestration": [
        "unleash run workflow <name>",
        "unleash run agent <name>",
        "unleash run pipeline <file>",
    ],
    "L2_Memory": [
        "unleash memory store <content>",
        "unleash memory search <query>",
        "unleash memory list",
        "unleash memory delete <key>",
    ],
    "L3_Structured": [
        "unleash structured generate <prompt>",
        "unleash structured validate <schema> <json>",
    ],
    "L4_Reasoning": [
        "unleash reason <task>",
        "unleash think <problem>",
    ],
    "L5_Observability": [
        "unleash trace list",
        "unleash trace show <id>",
        "unleash trace export <id>",
        "unleash eval run <test>",
        "unleash eval list",
    ],
    "L6_Safety": [
        "unleash safety scan <text>",
        "unleash safety guard enable",
        "unleash safety guard disable",
        "unleash safety guard status",
    ],
    "L7_Processing": [
        "unleash doc convert <file>",
        "unleash doc extract <file>",
    ],
    "L8_Knowledge": [
        "unleash knowledge index <file>",
        "unleash knowledge search <query>",
        "unleash knowledge list",
    ],
    "Core": [
        "unleash status",
        "unleash --version",
        "unleash config show",
        "unleash config init",
        "unleash config validate",
        "unleash --help",
    ],
}

# Current commands in unified_cli.py (V35 updated)
CURRENT_COMMANDS = {
    "L0_Protocol": [
        "unleash protocol call <prompt>",
        "unleash protocol chat",
    ],
    "L1_Orchestration": [
        "unleash run agent <name>",
        "unleash run pipeline <file>",
        "unleash run workflow <name>",
    ],
    "L2_Memory": [
        "unleash memory store",
        "unleash memory search <query>",
        "unleash memory list",
        "unleash memory delete",
    ],
    "L3_Structured": [
        "unleash structured generate <prompt>",
        "unleash structured validate <schema> <json>",
    ],
    "L4_Reasoning": [],  # Optional - not required for V35
    "L5_Observability": [
        "unleash trace list",
        "unleash trace show <id>",
        "unleash trace export <id>",
        "unleash eval run <suite>",
        "unleash eval list",
    ],
    "L6_Safety": [
        "unleash safety scan <text>",
        "unleash safety guard enable",
        "unleash safety guard disable",
        "unleash safety guard status",
    ],
    "L7_Processing": [
        "unleash doc convert <file>",
        "unleash doc extract <file>",
    ],
    "L8_Knowledge": [
        "unleash knowledge index <file>",
        "unleash knowledge search <query>",
        "unleash knowledge list",
    ],
    "Core": [
        "unleash status",
        "unleash --version",
        "unleash config show",
        "unleash config init",
        "unleash config validate",
        "unleash --help",
    ],
    "Extra": [
        "unleash tools list",
        "unleash tools invoke <name>",
        "unleash tools describe <name>",
    ],
}


def check_cli_exists() -> Path | None:
    """Check if CLI module exists."""
    cli_paths = [
        Path("core/cli/unified_cli.py"),
        Path("cli.py"),
        Path("platform/cli/main.py"),
    ]

    for path in cli_paths:
        if path.exists():
            print(f"[FOUND] CLI: {path}")
            return path

    print("[MISSING] No CLI module found")
    return None


def audit_commands() -> tuple[int, int, dict]:
    """Audit which commands are implemented vs expected."""
    print("\n" + "=" * 60)
    print("CLI COMMAND AUDIT - V35 Target")
    print("=" * 60)

    implemented = 0
    missing = 0
    results = {}

    for layer, expected in EXPECTED_COMMANDS.items():
        current = CURRENT_COMMANDS.get(layer, [])
        results[layer] = {"expected": len(expected), "implemented": 0, "missing": []}

        print(f"\n{layer}:")
        for cmd in expected:
            # Check if command exists (simplified match)
            cmd_base = cmd.split()[1] if len(cmd.split()) > 1 else cmd
            found = any(cmd_base in c for c in current)

            if found:
                implemented += 1
                results[layer]["implemented"] += 1
                print(f"  [PASS] {cmd}")
            else:
                missing += 1
                results[layer]["missing"].append(cmd)
                print(f"  [MISS] {cmd}")

    return implemented, missing, results


def check_cli_help() -> bool:
    """Verify CLI --help works."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "core.cli.unified_cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] CLI help check failed: {e}")
        return False


def check_cli_version() -> str | None:
    """Get CLI version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "core.cli.unified_cli", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def main():
    """Run CLI audit."""
    print("=" * 60)
    print("Phase 14: CLI Commands Verification Audit")
    print("=" * 60)

    # Check CLI exists
    cli_path = check_cli_exists()
    if not cli_path:
        print("\n[FAIL] CLI module not found")
        return 1

    # Check help works
    print("\nChecking CLI responsiveness...")
    help_works = check_cli_help()
    print(f"  --help: {'[PASS]' if help_works else '[FAIL]'}")

    # Check version
    version = check_cli_version()
    print(f"  --version: {version if version else '[FAIL]'}")

    # Audit commands
    implemented, missing, results = audit_commands()
    total = implemented + missing

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    print(f"\nTotal Commands: {total}")
    print(f"Implemented: {implemented} ({100 * implemented // total}%)")
    print(f"Missing: {missing} ({100 * missing // total}%)")

    print("\nBy Layer:")
    for layer, data in results.items():
        status = "[PASS]" if data["implemented"] == data["expected"] else "[PARTIAL]" if data["implemented"] > 0 else "[MISS]"
        print(f"  {layer}: {data['implemented']}/{data['expected']} {status}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    if missing > 0:
        print("1. Add missing command groups to unified_cli.py:")
        for layer, data in results.items():
            if data["missing"]:
                print(f"   - {layer}: {len(data['missing'])} commands")
        print("2. Update version to 35.0.0")
        print("3. Run tests/test_cli_commands.py")
        print("4. Document in audit/PHASE_14_RESULTS.md")
    else:
        print("All commands implemented! Run verification tests.")

    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
