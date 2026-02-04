"""
Root-level conftest for platform package.

Ensures sys.path is configured correctly BEFORE pytest collects test modules.
With --import-mode=importlib, test file imports use standard Python resolution,
so platform/ must be first in sys.path for 'from core.xxx' to resolve correctly.
"""

import sys
import os
from pathlib import Path

# CRITICAL: Fix platform namespace collision BEFORE any other imports.
_platform_pkg_dir = str(Path(__file__).parent)
_unleash_root = str(Path(__file__).parent.parent)

# Temporarily remove paths that could resolve to our 'platform' package
_original_path = sys.path[:]
sys.path = [p for p in sys.path if os.path.normpath(p) not in (
    os.path.normpath(_platform_pkg_dir),
    os.path.normpath(_unleash_root),
)]

# Force-load stdlib platform module
import importlib
if 'platform' in sys.modules:
    _platform_mod = sys.modules['platform']
    if not hasattr(_platform_mod, 'python_version'):
        del sys.modules['platform']
        import platform  # noqa: F811
else:
    import platform  # noqa: F811

# Restore paths
sys.path = _original_path

# Ensure platform dir is FIRST so 'from core.xxx' resolves to platform/core/
if _platform_pkg_dir not in sys.path:
    sys.path.insert(0, _platform_pkg_dir)
if _unleash_root not in sys.path:
    sys.path.insert(1, _unleash_root)
