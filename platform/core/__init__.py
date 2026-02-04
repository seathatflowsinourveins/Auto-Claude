"""
UAP Core Package - Lazy-loading foundation modules.

V45: Converted to lazy loading to prevent cascade import failures.
Submodules are loaded on demand via __getattr__.

For the full eager-loading version, see __init__eager.py.
"""

import importlib
import sys

# Module-level docstring preserved for discovery
__version__ = "45.0.0"

# Map of name -> (module_path, attribute_name) for lazy resolution
# This allows `from core import X` to work without eagerly loading everything
_LAZY_IMPORTS = {}


def _populate_lazy_imports():
    """Build the lazy import map from the eager init's __all__ and imports."""
    # Rather than maintaining a massive mapping, we use a fallback approach:
    # When an attribute is requested, we try to find it in submodules.
    pass


def __getattr__(name):
    """Lazy-load attributes from submodules on demand."""
    # Try to import from the eager init module
    try:
        eager = importlib.import_module("core.__init__eager")
        if hasattr(eager, name):
            val = getattr(eager, name)
            # Cache it on this module for future access
            globals()[name] = val
            return val
    except (ImportError, AttributeError):
        pass

    raise AttributeError(f"module 'core' has no attribute {name!r}")


# Expose __all__ from eager init lazily
def _get_all():
    try:
        eager = importlib.import_module("core.__init__eager")
        return getattr(eager, "__all__", [])
    except ImportError:
        return []


# Don't trigger eager loading at import time
# __all__ will be resolved on demand if needed
