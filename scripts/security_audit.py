#!/usr/bin/env python3
"""
Security Audit for Production Deployment
Phase 15: V35 Production Readiness

Scans codebase for hardcoded secrets and validates environment variables.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns that may indicate hardcoded secrets
SENSITIVE_PATTERNS = [
    (r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']', "Hardcoded API key"),
    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
    (r'secret\s*=\s*["\'][^"\']{10,}["\']', "Hardcoded secret"),
    (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key pattern"),
    (r'sk-ant-[a-zA-Z0-9-]{20,}', "Anthropic API key pattern"),
    (r'token\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']', "Hardcoded token"),
    (r'private[_-]?key\s*=', "Private key reference"),
    (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "Private key content"),
]

# Environment variables required for production
REQUIRED_ENV_VARS = [
    "ANTHROPIC_API_KEY",
]

# Optional but recommended environment variables
OPTIONAL_ENV_VARS = [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "MEM0_API_KEY",
    "FIRECRAWL_API_KEY",
]

# Directories to skip during scanning
SKIP_DIRS = {
    '.venv', 'venv', 'node_modules', '__pycache__', '.git',
    'dist', 'build', '.eggs', '*.egg-info', 'site-packages',
}

# File extensions to scan
SCAN_EXTENSIONS = {'.py', '.json', '.yaml', '.yml', '.env', '.toml', '.ini', '.cfg'}


def should_skip_path(filepath: Path) -> bool:
    """Check if path should be skipped."""
    for part in filepath.parts:
        if part in SKIP_DIRS or part.endswith('.egg-info'):
            return True
    return False


def scan_file(filepath: Path) -> List[Tuple[str, int, str]]:
    """Scan a file for sensitive patterns."""
    issues = []
    try:
        content = filepath.read_text(errors='ignore')
        lines = content.split('\n')

        for pattern, description in SENSITIVE_PATTERNS:
            for i, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('//'):
                    continue
                # Skip template/example patterns
                if 'your-' in line.lower() or 'xxx' in line.lower() or 'example' in line.lower():
                    continue
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append((str(filepath), i, description))
    except Exception:
        pass
    return issues


def audit_codebase(root: Path) -> Tuple[int, List[Tuple[str, int, str]]]:
    """Audit entire codebase for security issues."""
    print("=" * 60)
    print("SECURITY AUDIT - V35 Production Deployment")
    print("=" * 60)

    issues = []
    files_scanned = 0

    for ext in SCAN_EXTENSIONS:
        pattern = f'*{ext}'
        for filepath in root.rglob(pattern):
            if should_skip_path(filepath):
                continue
            files_scanned += 1
            issues.extend(scan_file(filepath))

    print(f"\nScanned {files_scanned} files")

    if issues:
        print(f"\n[WARN] Found {len(issues)} potential security issues:")
        for filepath, line, desc in issues[:20]:  # Show first 20
            rel_path = Path(filepath).relative_to(root) if filepath.startswith(str(root)) else filepath
            print(f"  {rel_path}:{line} - {desc}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\n[PASS] No hardcoded secrets found")

    return len(issues), issues


def check_environment_variables():
    """Check required environment variables."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 60)

    missing_required = 0
    missing_optional = 0

    print("\nRequired:")
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            # Mask the value
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print(f"  [PASS] {var} = {masked}")
        else:
            print(f"  [MISS] {var} is NOT set")
            missing_required += 1

    print("\nOptional:")
    for var in OPTIONAL_ENV_VARS:
        value = os.getenv(var)
        if value:
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print(f"  [SET]  {var} = {masked}")
        else:
            print(f"  [----] {var} not set")
            missing_optional += 1

    return missing_required


def check_file_permissions(root: Path):
    """Check file permissions for sensitive files."""
    print("\n" + "=" * 60)
    print("FILE PERMISSIONS CHECK")
    print("=" * 60)

    sensitive_patterns = ['*.env', '*.pem', '*.key', '*credentials*', '*secret*']
    issues = 0

    for pattern in sensitive_patterns:
        for filepath in root.rglob(pattern):
            if should_skip_path(filepath):
                continue
            # On Windows, just note the existence
            if sys.platform == 'win32':
                print(f"  [INFO] Sensitive file found: {filepath.relative_to(root)}")
            else:
                mode = filepath.stat().st_mode & 0o777
                if mode & 0o077:  # Group/other readable
                    print(f"  [WARN] {filepath.relative_to(root)} has permissive mode {oct(mode)}")
                    issues += 1
                else:
                    print(f"  [PASS] {filepath.relative_to(root)} permissions OK")

    if issues == 0:
        print("  [PASS] No permission issues found")

    return issues


def main():
    """Run security audit."""
    root = Path(__file__).parent.parent

    # Run all checks
    secret_issues, _ = audit_codebase(root)
    env_issues = check_environment_variables()
    perm_issues = check_file_permissions(root)

    # Summary
    print("\n" + "=" * 60)
    print("SECURITY AUDIT SUMMARY")
    print("=" * 60)

    total_issues = secret_issues + env_issues + perm_issues

    if total_issues == 0:
        print("\n[PASS] Security audit passed!")
        print("  - No hardcoded secrets found")
        print("  - Required environment variables set")
        print("  - File permissions OK")
        return 0
    else:
        print(f"\n[WARN] Found {total_issues} issue(s) to address:")
        if secret_issues:
            print(f"  - {secret_issues} potential hardcoded secret(s)")
        if env_issues:
            print(f"  - {env_issues} missing required environment variable(s)")
        if perm_issues:
            print(f"  - {perm_issues} file permission issue(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
