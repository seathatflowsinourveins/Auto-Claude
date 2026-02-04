"""Root conftest.py - fixes 'platform' stdlib namespace collision.

The platform/ directory in this project shadows Python's stdlib 'platform'
module. This conftest ensures stdlib platform is loaded into sys.modules
before pytest descends into platform/tests/.
"""
import sys
import platform as _stdlib_platform  # noqa: F401

# Ensure stdlib platform stays in sys.modules
assert hasattr(_stdlib_platform, 'python_version'), (
    "stdlib platform module not loaded correctly"
)
# Pin it so nothing can shadow it
sys.modules['platform'] = _stdlib_platform
