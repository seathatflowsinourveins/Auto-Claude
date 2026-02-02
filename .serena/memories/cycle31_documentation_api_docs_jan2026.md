# Cycle 31: Documentation & API Docs Patterns (January 2026)

**Research Date**: 2026-01-25
**Focus**: MkDocs, Sphinx, docstrings, OpenAPI, FastAPI autodocs

---

## 1. Documentation Quadrant Framework

From 2026 best practices - four types of documentation:

```
┌─────────────────────────────────────────────────────────────┐
│                  DOCUMENTATION QUADRANT                      │
├──────────────────────────┬──────────────────────────────────┤
│       LEARNING           │         INFORMATION              │
│  ────────────────────    │    ──────────────────────        │
│  Tutorials               │    Reference                     │
│  • Get started guides    │    • API documentation           │
│  • Step-by-step          │    • Configuration options       │
│  • Learning-oriented     │    • Technical specs             │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│     UNDERSTANDING        │         GOAL-ORIENTED            │
│  ────────────────────    │    ──────────────────────        │
│  Explanation             │    How-to Guides                 │
│  • Background concepts   │    • Solve specific problems     │
│  • Design decisions      │    • Task-oriented               │
│  • Architecture          │    • Practical steps             │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 2. MkDocs + Material Theme (2026 Standard)

### Production Configuration
```yaml
# mkdocs.yml
site_name: My Project
site_url: https://docs.example.com
repo_url: https://github.com/org/project

theme:
  name: material
  features:
    - navigation.instant      # XHR-based navigation (SPA-like)
    - navigation.tabs         # Top-level sections as tabs
    - navigation.sections     # Expandable sections
    - navigation.expand       # Auto-expand navigation
    - navigation.top          # Back-to-top button
    - search.suggest          # Search suggestions
    - search.highlight        # Highlight search terms
    - content.code.copy       # Copy button for code blocks
    - content.code.annotate   # Code annotations
  palette:
    - scheme: default
      primary: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true

markdown_extensions:
  - admonition            # Note/warning/tip blocks
  - pymdownx.details      # Collapsible sections
  - pymdownx.superfences  # Fenced code blocks
  - pymdownx.tabbed:      # Tabbed content
      alternate_style: true
  - pymdownx.highlight:   # Syntax highlighting
      anchor_linenums: true
  - attr_list             # Add HTML attributes
  - md_in_html            # Markdown inside HTML

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
  - API Reference: reference/
  - Changelog: changelog.md
```

### GitHub Actions Deployment
```yaml
# .github/workflows/docs.yml
name: Deploy Docs
on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: pip install mkdocs-material mkdocstrings[python]
      
      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
```

---

## 3. Docstring Styles Comparison

### Google Style (Recommended)
```python
def fetch_user(user_id: int, include_profile: bool = False) -> User:
    """Fetch a user by their ID.

    Retrieves the user from the database with optional profile data.
    
    Args:
        user_id: The unique identifier for the user.
        include_profile: Whether to include extended profile data.
            Defaults to False.

    Returns:
        The User object with the requested data.

    Raises:
        UserNotFoundError: If no user exists with the given ID.
        DatabaseError: If the database connection fails.

    Example:
        >>> user = fetch_user(123)
        >>> print(user.name)
        'John Doe'
        
        >>> user = fetch_user(123, include_profile=True)
        >>> print(user.profile.bio)
        'Software developer'
    """
```

### NumPy Style
```python
def calculate_statistics(data: np.ndarray) -> dict[str, float]:
    """
    Calculate descriptive statistics for the input array.

    Parameters
    ----------
    data : np.ndarray
        Input array of numerical values.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'mean', 'std', 'min', 'max' keys.

    See Also
    --------
    numpy.mean : Compute the arithmetic mean.
    numpy.std : Compute the standard deviation.

    Notes
    -----
    Uses Bessel's correction (ddof=1) for standard deviation.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> stats = calculate_statistics(data)
    >>> stats['mean']
    3.0
    """
```

### Style Selection Guide
| Style | Best For | Tool Support |
|-------|----------|--------------|
| Google | General Python projects | mkdocstrings, Sphinx |
| NumPy | Scientific/data projects | mkdocstrings, Sphinx |
| Sphinx (reST) | Legacy Sphinx projects | Sphinx native |

**Rule**: Pick one style, enforce with linting, stay consistent.

---

## 4. mkdocstrings Configuration

### Auto-Generate API Docs
```yaml
# mkdocs.yml
plugins:
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src]
          options:
            # Docstring rendering
            docstring_style: google
            docstring_section_style: spacy
            merge_init_into_class: true
            
            # Display options
            show_source: true
            show_root_heading: true
            show_root_full_path: false
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            
            # Member options
            members_order: source
            group_by_category: true
            show_if_no_docstring: false
            
            # Signature options
            separate_signature: true
            show_signature_annotations: true
```

### In Markdown Files
```markdown
# API Reference

## User Module

::: mypackage.models.user
    options:
      show_root_heading: true
      members:
        - User
        - UserProfile

## Utilities

::: mypackage.utils
    options:
      show_source: false
```

---

## 5. FastAPI Automatic OpenAPI Documentation

### Built-in Documentation
```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="API for managing users and orders",
    version="1.0.0",
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc
    openapi_url="/openapi.json" # OpenAPI schema
)

# Schema is auto-generated from routes + Pydantic models
```

### Enhanced Documentation with Metadata
```python
from fastapi import FastAPI, Path, Query
from pydantic import BaseModel, Field

class User(BaseModel):
    """User model for API responses."""
    id: int = Field(..., description="Unique user identifier", example=123)
    name: str = Field(..., description="User's full name", example="John Doe")
    email: str = Field(..., description="User's email address", example="john@example.com")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ]
        }
    }

@app.get(
    "/users/{user_id}",
    response_model=User,
    summary="Get a user by ID",
    description="Retrieve a single user by their unique identifier.",
    response_description="The user object",
    tags=["Users"],
    responses={
        404: {"description": "User not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_user(
    user_id: int = Path(..., description="The ID of the user to retrieve", ge=1),
    include_profile: bool = Query(False, description="Include extended profile data")
) -> User:
    """
    Get a user by their ID.
    
    - **user_id**: Unique identifier for the user
    - **include_profile**: Optional flag to include profile data
    """
    return User(id=user_id, name="John Doe", email="john@example.com")
```

### Export OpenAPI Schema
```python
import json
from fastapi.openapi.utils import get_openapi

# Generate and save OpenAPI schema
def export_openapi():
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    with open("openapi.json", "w") as f:
        json.dump(schema, f, indent=2)
```

---

## 6. Sphinx Configuration (Legacy/Complex Projects)

### conf.py Setup
```python
# docs/conf.py
project = 'My Project'
copyright = '2026, My Team'
author = 'My Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # Google/NumPy docstrings
    'sphinx.ext.viewcode',      # Link to source
    'sphinx.ext.intersphinx',   # Cross-project links
    'sphinx_autodoc_typehints', # Type hint support
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Intersphinx (link to other docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

html_theme = 'furo'  # Modern theme
```

---

## 7. Type Hints + Docstrings Integration

### Modern Pattern (2026)
```python
def process_data(
    items: list[dict[str, Any]],
    *,
    max_items: int = 100,
    filter_fn: Callable[[dict], bool] | None = None,
) -> tuple[list[dict], int]:
    """Process a list of data items with optional filtering.

    Type information is in the signature, so docstring focuses on semantics.

    Args:
        items: Raw data items to process.
        max_items: Maximum number of items to return.
        filter_fn: Optional predicate to filter items.

    Returns:
        A tuple of (processed items, total count before limiting).

    Note:
        Type hints provide the type info - docstring explains behavior.
    """
```

### Key Insight from 2026 Survey
> "Modern Python (3.5+) allows you to declare types in the function signature. 
> This creates a single source of truth, so docstrings focus on SEMANTICS, 
> not type declarations."

---

## 8. Documentation CI/CD Pipeline

### Complete Workflow
```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install mkdocs-material mkdocstrings[python]
          pip install -e ".[dev]"
      
      - name: Build docs
        run: mkdocs build --strict
      
      - name: Check links
        run: |
          pip install linkchecker
          linkchecker site/
  
  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - run: pip install mkdocs-material mkdocstrings[python]
      - run: mkdocs gh-deploy --force
```

---

## 9. README Best Practices

### Structure Template
```markdown
# Project Name

[![PyPI](https://img.shields.io/pypi/v/project)](https://pypi.org/project/project/)
[![Tests](https://github.com/org/project/workflows/tests/badge.svg)](https://github.com/org/project/actions)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://org.github.io/project/)

One-line description of what the project does.

## Features

- ✅ Feature one
- ✅ Feature two
- ✅ Feature three

## Installation

\`\`\`bash
pip install project
\`\`\`

## Quick Start

\`\`\`python
from project import Thing

thing = Thing()
result = thing.do_something()
\`\`\`

## Documentation

Full documentation at [https://org.github.io/project/](https://org.github.io/project/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE)
```

---

## 10. Tool Comparison Summary

| Tool | Best For | Output | Learning Curve |
|------|----------|--------|----------------|
| **MkDocs + Material** | Modern projects | Static HTML | Low |
| **Sphinx** | Complex/legacy | Static HTML | Medium |
| **FastAPI built-in** | API docs | Swagger/ReDoc | None |
| **pdoc** | Quick API docs | Static HTML | Very Low |

### Recommendation (2026)
1. **Project docs**: MkDocs + Material + mkdocstrings
2. **API docs**: FastAPI auto-generation + Bump.sh hosting
3. **Docstring style**: Google (best tooling support)
4. **Hosting**: GitHub Pages (free) or Read the Docs

---

*Cycle 31 Complete | Documentation & API Docs Patterns*
