# Cycle 29: Packaging & Distribution Patterns (January 25, 2026)

**Focus**: Modern Python packaging, versioning automation, PyPI trusted publishing

---

## 1. pyproject.toml (PEP 621 Standard)

### Complete pyproject.toml Template
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mypackage"
version = "0.1.0"
description = "Production Python package"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Developer", email = "dev@example.com"}
]
keywords = ["sdk", "api", "automation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Typing :: Typed",
]

dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "mypy",
    "ruff",
]
docs = ["mkdocs-material"]

[project.scripts]
myapp = "mypackage.cli:main"

[project.urls]
Homepage = "https://github.com/org/mypackage"
Documentation = "https://mypackage.readthedocs.io"
Changelog = "https://github.com/org/mypackage/blob/main/CHANGELOG.md"

# Tool configurations
[tool.hatch.version]
path = "src/mypackage/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --cov=mypackage"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.ruff]
target-version = "py311"
line-length = 88
select = ["E", "F", "I", "UP", "B", "C4"]
```

### Build Backend Comparison (2026)

| Backend | Best For | Key Features |
|---------|----------|--------------|
| **Hatchling** | New projects | Fast, extensible, dynamic versioning |
| **setuptools** | Existing projects | Mature, widely compatible |
| **Poetry-core** | Poetry users | Lock files, dependency groups |
| **Flit-core** | Simple packages | Minimal config, pure Python |
| **PDM-backend** | PDM users | PEP 582 support |

### Dynamic Versioning (Single Source of Truth)
```toml
# Option 1: Hatchling from __init__.py
[tool.hatch.version]
path = "src/mypackage/__init__.py"
# __init__.py contains: __version__ = "1.2.3"

# Option 2: setuptools-scm (from git tags)
[build-system]
requires = ["setuptools>=64", "setuptools-scm>=8"]

[tool.setuptools_scm]
version_scheme = "guess-next-dev"
```

---

## 2. Semantic Versioning & Automation

### Version Format (PEP 440)
```
MAJOR.MINOR.PATCH[pre][post][dev]

Examples:
1.0.0        # Release
1.0.1        # Patch (backwards compatible bug fix)
1.1.0        # Minor (backwards compatible feature)
2.0.0        # Major (breaking change)
1.0.0a1      # Alpha pre-release
1.0.0b2      # Beta pre-release
1.0.0rc1     # Release candidate
1.0.0.post1  # Post-release
1.0.0.dev1   # Development release
```

### python-semantic-release (Automated Versioning)
```toml
# pyproject.toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
version_variables = ["src/mypackage/__init__.py:__version__"]
branch = "main"
build_command = "python -m build"
commit_message = "chore(release): {version}"
tag_format = "v{version}"

[tool.semantic_release.changelog]
template_dir = "templates"
changelog_file = "CHANGELOG.md"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "perf", "refactor", "docs", "chore"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]
```

### Conventional Commits (Required for Auto-Versioning)
```
feat: add user authentication       → MINOR bump (1.1.0)
fix: resolve login race condition   → PATCH bump (1.0.1)
feat!: redesign API                 → MAJOR bump (2.0.0)
BREAKING CHANGE: remove old API     → MAJOR bump

Format: <type>(<scope>): <description>
```

### GitHub Actions for Semantic Release
```yaml
name: Release

on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    concurrency: release
    permissions:
      id-token: write
      contents: write
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for changelog
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - run: pip install python-semantic-release
      
      - name: Semantic Release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: semantic-release version --changelog --push --tag
```

---

## 3. PyPI Trusted Publishing (OIDC)

### Why Trusted Publishing?
- **No API tokens** to manage or rotate
- **No secrets** stored in GitHub
- Short-lived credentials (auto-expire)
- Stronger identity verification
- Audit trail via OIDC claims

### Configure PyPI Trusted Publisher
1. Go to PyPI → Your Project → Publishing
2. Add "GitHub Actions" as trusted publisher:
   - Owner: `your-org`
   - Repository: `your-repo`
   - Workflow: `publish.yml`
   - Environment: `pypi` (optional but recommended)

### GitHub Actions Workflow (Trusted Publishing)
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - name: Build package
        run: |
          pip install build
          python -m build
      
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi  # Optional: adds approval gate
    permissions:
      id-token: write  # REQUIRED for trusted publishing
    
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      
      - uses: pypa/gh-action-pypi-publish@release/v1
        # No password/token needed - uses OIDC!
```

### TestPyPI First (Recommended)
```yaml
# Add separate job for TestPyPI
publish-test:
  needs: build
  runs-on: ubuntu-latest
  permissions:
    id-token: write
  steps:
    - uses: actions/download-artifact@v4
      with:
        name: dist
        path: dist/
    
    - uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://test.pypi.org/legacy/
```

---

## 4. Package Structure Best Practices

### src Layout (Recommended)
```
mypackage/
├── src/
│   └── mypackage/
│       ├── __init__.py      # __version__ = "1.0.0"
│       ├── py.typed         # PEP 561 marker
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── conftest.py
│   └── test_core.py
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── LICENSE
```

### Why src/ Layout?
1. **Prevents accidental imports** from source during testing
2. **Forces installed package testing** (catches packaging errors)
3. **Clear separation** of source vs tests vs config
4. **Industry standard** for 2026

### py.typed Marker (PEP 561)
```python
# Create empty src/mypackage/py.typed file
# Tells type checkers this package has type annotations

# Include in package
[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
artifacts = ["src/mypackage/py.typed"]
```

---

## 5. Dependency Management

### Version Specifiers (PEP 508)
```toml
dependencies = [
    "httpx>=0.25.0,<1.0",      # Range (recommended)
    "pydantic>=2.0",           # Minimum only
    "numpy~=1.24.0",           # Compatible release (~=1.24.0 means >=1.24.0,<1.25.0)
    "requests==2.31.0",        # Exact pin (AVOID in libraries)
]
```

### Libraries vs Applications
```toml
# LIBRARY: Wide ranges (let users resolve)
dependencies = ["httpx>=0.25"]

# APPLICATION: Lock files for reproducibility
# Use poetry.lock, pdm.lock, or requirements.txt
pip freeze > requirements.txt
```

### Optional Dependencies (Extras)
```toml
[project.optional-dependencies]
dev = ["pytest", "mypy", "ruff"]
docs = ["mkdocs-material", "mkdocstrings"]
all = ["mypackage[dev,docs]"]

# Install: pip install mypackage[dev]
```

---

## 6. Build & Distribution Commands

### Build Commands
```bash
# Build with build (recommended)
pip install build
python -m build           # Creates dist/*.whl and dist/*.tar.gz

# Build with hatch
pip install hatch
hatch build

# Build with poetry
poetry build
```

### Local Testing
```bash
# Install in editable mode
pip install -e ".[dev]"

# Test the built package
pip install dist/mypackage-1.0.0-py3-none-any.whl
```

### Upload Commands (Manual)
```bash
# Upload to TestPyPI first
pip install twine
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

---

## Key Takeaways

| Area | Best Practice |
|------|---------------|
| Config | Single pyproject.toml (PEP 621) |
| Backend | Hatchling for new, setuptools for existing |
| Layout | src/ layout with py.typed |
| Versioning | python-semantic-release + Conventional Commits |
| Publishing | Trusted Publishing (OIDC) - no tokens |
| Dependencies | Ranges for libraries, locks for apps |

## Anti-Patterns to Avoid

1. **setup.py only** - Migrate to pyproject.toml
2. **Flat layout** - Use src/ layout
3. **Manual versioning** - Automate with semantic-release
4. **API tokens in secrets** - Use Trusted Publishing
5. **Exact pins in libraries** - Use version ranges

---

*Next: Cycle 30 - Type System & Static Analysis Patterns*
