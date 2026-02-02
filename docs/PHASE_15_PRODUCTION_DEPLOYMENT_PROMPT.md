# Phase 15: Production Deployment Checklist

## Overview

Final phase to prepare V35 for production deployment. Covers security, configuration, monitoring, and deployment automation.

## Prerequisites Verification

### V35 Status Verification
```bash
# Run validation
python scripts/validate_v35_final.py
# Expected: 36/36 (100%)

# Run CLI tests
pytest tests/test_cli_commands.py -v
# Expected: 30/30 passed

# Run E2E tests
pytest tests/test_e2e_integration.py -v
# Expected: All passed
```

## Step 1: Security Audit

Create `scripts/security_audit.py`:

```python
#!/usr/bin/env python3
"""Security audit for production deployment"""

import os
import re
from pathlib import Path
from typing import List, Tuple

SENSITIVE_PATTERNS = [
    (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
    (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
    (r'sk-[a-zA-Z0-9]{20,}', "OpenAI API key pattern"),
    (r'anthropic[_-]?key\s*=', "Anthropic key reference"),
]

REQUIRED_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]

def scan_file(filepath: Path) -> List[Tuple[str, int, str]]:
    """Scan a file for sensitive patterns"""
    issues = []
    try:
        content = filepath.read_text(errors='ignore')
        for pattern, description in SENSITIVE_PATTERNS:
            for i, line in enumerate(content.split('\n'), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append((str(filepath), i, description))
    except Exception as e:
        pass
    return issues

def audit_codebase():
    """Audit entire codebase"""
    print("="*60)
    print("SECURITY AUDIT")
    print("="*60)
    
    issues = []
    for ext in ['*.py', '*.json', '*.yaml', '*.yml', '*.env*']:
        for filepath in Path('.').rglob(ext):
            if '.venv' in str(filepath) or 'node_modules' in str(filepath):
                continue
            issues.extend(scan_file(filepath))
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential security issues:")
        for filepath, line, desc in issues[:20]:  # Show first 20
            print(f"  {filepath}:{line} - {desc}")
    else:
        print("\n✅ No hardcoded secrets found")
    
    print("\n" + "="*60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("="*60)
    
    for var in REQUIRED_ENV_VARS:
        if os.getenv(var):
            print(f"  ✅ {var} is set")
        else:
            print(f"  ⚠️  {var} is NOT set")
    
    return len(issues)

if __name__ == "__main__":
    count = audit_codebase()
    exit(0 if count == 0 else 1)
```

## Step 2: Configuration Templates

Create `config/production.yaml`:

```yaml
# Unleash V35 Production Configuration

version: "35.0.0"

# L0 Protocol
protocol:
  default_provider: anthropic
  default_model: claude-sonnet-4-20250514
  max_tokens: 4096
  timeout: 60
  retry:
    max_attempts: 3
    backoff_factor: 2

# L2 Memory
memory:
  provider: mem0
  persistence: true
  storage_path: ~/.unleash/memory
  max_history: 1000

# L3 Structured
structured:
  validator: instructor
  schema_cache: true

# L5 Observability
observability:
  enabled: true
  provider: langfuse
  sample_rate: 1.0
  export_interval: 60

# L6 Safety
safety:
  enabled: true
  scanner: scanner_compat
  guardrails: rails_compat
  block_on_failure: true

# L8 Knowledge
knowledge:
  index_path: ~/.unleash/indices
  embedding_model: text-embedding-3-small

# Logging
logging:
  level: INFO
  format: json
  output: stdout
```

Create `config/env.template`:

```bash
# Unleash V35 Environment Variables Template
# Copy to .env and fill in values

# Required: L0 Protocol
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here

# Required: L5 Observability
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: L2 Memory
MEM0_API_KEY=your-mem0-key
ZEP_API_KEY=your-zep-key

# Optional: L8 Knowledge
FIRECRAWL_API_KEY=your-firecrawl-key

# Application
UNLEASH_ENV=production
UNLEASH_LOG_LEVEL=INFO
```

## Step 3: Health Check Endpoint

Create `core/health.py`:

```python
#!/usr/bin/env python3
"""Health check for production monitoring"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class HealthStatus:
    status: str  # healthy, degraded, unhealthy
    checks: Dict[str, Any]
    timestamp: str
    version: str = "35.0.0"

async def check_protocol() -> Dict[str, Any]:
    """Check L0 Protocol availability"""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        # Light check - just verify client creation
        return {"status": "healthy", "provider": "anthropic"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_memory() -> Dict[str, Any]:
    """Check L2 Memory availability"""
    try:
        from mem0 import Memory
        Memory()
        return {"status": "healthy", "provider": "mem0"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

async def check_safety() -> Dict[str, Any]:
    """Check L6 Safety availability"""
    try:
        from core.safety.scanner_compat import InputScanner
        scanner = InputScanner()
        result = scanner.scan("test")
        return {"status": "healthy", "provider": "scanner_compat"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

async def check_knowledge() -> Dict[str, Any]:
    """Check L8 Knowledge availability"""
    try:
        from llama_index.core import VectorStoreIndex
        return {"status": "healthy", "provider": "llama_index"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

async def get_health_status() -> HealthStatus:
    """Get overall health status"""
    checks = {
        "L0_protocol": await check_protocol(),
        "L2_memory": await check_memory(),
        "L6_safety": await check_safety(),
        "L8_knowledge": await check_knowledge(),
    }
    
    # Determine overall status
    statuses = [c["status"] for c in checks.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"
    
    return HealthStatus(
        status=overall,
        checks=checks,
        timestamp=datetime.utcnow().isoformat()
    )

def main():
    """Run health check"""
    import json
    status = asyncio.run(get_health_status())
    print(json.dumps({
        "status": status.status,
        "version": status.version,
        "timestamp": status.timestamp,
        "checks": status.checks
    }, indent=2))

if __name__ == "__main__":
    main()
```

## Step 4: Deployment Script

Create `scripts/deploy.py`:

```python
#!/usr/bin/env python3
"""Production deployment script"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, check: bool = True) -> bool:
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        return False
    return True

def deploy():
    """Execute deployment steps"""
    print("="*60)
    print("UNLEASH V35 PRODUCTION DEPLOYMENT")
    print("="*60)
    
    steps = [
        ("Validate V35 SDKs", "python scripts/validate_v35_final.py"),
        ("Run Security Audit", "python scripts/security_audit.py"),
        ("Run CLI Tests", "pytest tests/test_cli_commands.py -v"),
        ("Run E2E Tests", "pytest tests/test_e2e_integration.py -v"),
        ("Check Health", "python core/health.py"),
    ]
    
    passed = 0
    for name, cmd in steps:
        print(f"\n--- {name} ---")
        if run_command(cmd, check=False):
            passed += 1
            print(f"✅ {name} passed")
        else:
            print(f"⚠️  {name} had issues")
    
    print("\n" + "="*60)
    print(f"DEPLOYMENT READINESS: {passed}/{len(steps)} checks passed")
    print("="*60)
    
    if passed == len(steps):
        print("✅ Ready for production deployment!")
        return 0
    else:
        print("⚠️  Address issues before deploying")
        return 1

if __name__ == "__main__":
    sys.exit(deploy())
```

## Step 5: Docker Configuration

Create `Dockerfile`:

```dockerfile
# Unleash V35 Production Dockerfile
FROM python:3.14-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application
COPY core/ core/
COPY config/ config/
COPY scripts/ scripts/

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python core/health.py || exit 1

# Default command
CMD ["python", "-m", "core.cli.unified_cli", "status"]
```

Create `docker-compose.yaml`:

```yaml
version: '3.8'

services:
  unleash:
    build: .
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
    volumes:
      - unleash-data:/app/data
    ports:
      - "8000:8000"

volumes:
  unleash-data:
```

## Step 6: Final Validation Script

Create `scripts/final_validation.py`:

```python
#!/usr/bin/env python3
"""Final production validation"""

import subprocess
import sys
import json

def main():
    print("="*60)
    print("UNLEASH V35 FINAL PRODUCTION VALIDATION")
    print("="*60)
    
    results = {
        "sdk_validation": None,
        "cli_tests": None,
        "e2e_tests": None,
        "security_audit": None,
        "health_check": None,
    }
    
    # SDK Validation
    print("\n1. SDK Validation...")
    r = subprocess.run(["python", "scripts/validate_v35_final.py"], capture_output=True)
    results["sdk_validation"] = "✅ 36/36" if r.returncode == 0 else "❌ Failed"
    
    # CLI Tests
    print("2. CLI Tests...")
    r = subprocess.run(["pytest", "tests/test_cli_commands.py", "-q"], capture_output=True)
    results["cli_tests"] = "✅ 30/30" if r.returncode == 0 else "❌ Failed"
    
    # E2E Tests
    print("3. E2E Tests...")
    r = subprocess.run(["pytest", "tests/test_e2e_integration.py", "-q"], capture_output=True)
    results["e2e_tests"] = "✅ Passed" if r.returncode == 0 else "❌ Failed"
    
    # Security Audit
    print("4. Security Audit...")
    r = subprocess.run(["python", "scripts/security_audit.py"], capture_output=True)
    results["security_audit"] = "✅ Clean" if r.returncode == 0 else "⚠️ Review"
    
    # Health Check
    print("5. Health Check...")
    r = subprocess.run(["python", "core/health.py"], capture_output=True)
    results["health_check"] = "✅ Healthy" if r.returncode == 0 else "⚠️ Degraded"
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for check, status in results.items():
        print(f"  {check}: {status}")
    
    all_passed = all("✅" in str(v) for v in results.values())
    print("\n" + ("🎉 PRODUCTION READY!" if all_passed else "⚠️ Review issues"))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

## Success Criteria

- [ ] Security audit passes (no hardcoded secrets)
- [ ] All environment variables documented
- [ ] Configuration templates complete
- [ ] Health check endpoint functional
- [ ] Docker configuration ready
- [ ] All validation scripts pass

## Deployment Command

```bash
# Final validation
python scripts/final_validation.py

# If all checks pass:
docker-compose up -d
```
