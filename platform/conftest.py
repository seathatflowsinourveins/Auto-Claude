"""
Root-level conftest for platform package.

Ensures sys.path is configured correctly BEFORE pytest collects test modules.
With --import-mode=importlib, test file imports use standard Python resolution,
so platform/ must be first in sys.path for 'from core.xxx' to resolve correctly.
"""

import sys
import os
from pathlib import Path

# Prevent Keras 3 / TensorFlow incompatibility crash in transformers
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

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
# Remove unleash_root first if present (it has a conflicting core/ package)
sys.path = [p for p in sys.path
            if os.path.normpath(p) != os.path.normpath(_unleash_root)]
if _platform_pkg_dir not in sys.path:
    sys.path.insert(0, _platform_pkg_dir)
# Add unleash root AFTER platform dir
sys.path.append(_unleash_root)

# Purge any cached 'core' module that points to the wrong location
# (unleash/core/ instead of platform/core/)
_platform_core = os.path.normpath(os.path.join(_platform_pkg_dir, 'core'))
if 'core' in sys.modules:
    _core_mod = sys.modules['core']
    _core_file = getattr(_core_mod, '__file__', '') or ''
    if _core_file and os.path.normpath(os.path.dirname(_core_file)) != _platform_core:
        # Wrong core package loaded - purge it and all submodules
        _to_remove = [k for k in sys.modules if k == 'core' or k.startswith('core.')]
        for k in _to_remove:
            del sys.modules[k]
