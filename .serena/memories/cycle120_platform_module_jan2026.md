# Python platform Module - Production Patterns (Cycle 120)

## Overview
The `platform` module provides access to underlying platform identifying data. Essential for cross-platform applications, system diagnostics, telemetry, and compatibility checks.

## Cross-Platform Functions

### Core System Information
```python
import platform

# OS name: 'Linux', 'Darwin', 'Windows', 'iOS', 'Android'
os_name = platform.system()

# OS release version
release = platform.release()  # e.g., '5.15.0-91-generic', '10.0.22621'

# OS version details
version = platform.version()  # e.g., '#101-Ubuntu SMP', '10.0.22621'

# Machine architecture
machine = platform.machine()  # e.g., 'x86_64', 'AMD64', 'arm64'

# Network hostname
hostname = platform.node()

# Processor name (often same as machine)
processor = platform.processor()  # e.g., 'Intel64 Family 6 Model 140'
```

### uname() - All-in-One
```python
import platform

# Returns namedtuple with all info
info = platform.uname()
print(info.system)     # 'Linux'
print(info.node)       # 'myhost'
print(info.release)    # '5.15.0-91-generic'
print(info.version)    # '#101-Ubuntu SMP'
print(info.machine)    # 'x86_64'
print(info.processor)  # 'x86_64' (resolved lazily)

# Python 3.14+: Clear cached values
platform.invalidate_caches()
```

### platform() - Human-Readable String
```python
import platform

# Full platform description
desc = platform.platform()
# 'Linux-5.15.0-91-generic-x86_64-with-glibc2.35'
# 'macOS-14.2.1-arm64-arm-64bit'
# 'Windows-10-10.0.22621-SP0'

# Terse version
terse = platform.platform(terse=True)
# 'Linux-5.15.0-91-generic-x86_64'

# With aliases (SunOS → Solaris)
aliased = platform.platform(aliased=True)
```

### Python Information
```python
import platform

# Python version
version = platform.python_version()        # '3.12.1'
version_tuple = platform.python_version_tuple()  # ('3', '12', '1')

# Implementation
impl = platform.python_implementation()    # 'CPython', 'PyPy', etc.

# Build info
build_no, build_date = platform.python_build()
# ('v3.12.1:2305ca5144', 'Dec 7 2023 22:03:25')

# Compiler
compiler = platform.python_compiler()
# 'GCC 11.4.0' or 'MSC v.1937 64 bit (AMD64)'

# SCM info
branch = platform.python_branch()          # 'main'
revision = platform.python_revision()      # 'abc123...'
```

### Architecture Detection
```python
import platform
import sys

# Get architecture (relies on 'file' command)
bits, linkage = platform.architecture()
# ('64bit', 'ELF') on Linux
# ('64bit', '') on Windows

# More reliable 64-bit check
is_64bit = sys.maxsize > 2**32
```

## Platform-Specific Functions

### Windows
```python
import platform

if platform.system() == 'Windows':
    # Detailed Windows info
    release, version, csd, ptype = platform.win32_ver()
    # ('10', '10.0.22621', 'SP0', 'Multiprocessor Free')
    
    # Windows edition (3.8+)
    edition = platform.win32_edition()  # 'Enterprise', 'Professional'
    
    # IoT check (3.8+)
    is_iot = platform.win32_is_iot()
```

### macOS
```python
import platform

if platform.system() == 'Darwin':
    release, versioninfo, machine = platform.mac_ver()
    # ('14.2.1', ('', '', ''), 'arm64')
    
    version, dev_stage, non_release = versioninfo
```

### Linux
```python
import platform

if platform.system() == 'Linux':
    # libc version
    lib, version = platform.libc_ver()
    # ('glibc', '2.35')
    
    # freedesktop os-release (3.10+)
    try:
        os_info = platform.freedesktop_os_release()
        # {'NAME': 'Ubuntu', 'VERSION': '22.04.3 LTS', 
        #  'ID': 'ubuntu', 'ID_LIKE': 'debian', ...}
        
        distro_id = os_info['ID']           # 'ubuntu'
        distro_name = os_info['PRETTY_NAME']  # 'Ubuntu 22.04.3 LTS'
        
        # Get distribution family
        ids = [os_info['ID']]
        if 'ID_LIKE' in os_info:
            ids.extend(os_info['ID_LIKE'].split())
        # ['ubuntu', 'debian'] or ['fedora', 'rhel']
        
    except OSError:
        pass  # os-release not available
```

### iOS (3.13+)
```python
import platform

if platform.system() in ('iOS', 'iPadOS'):
    ios_info = platform.ios_ver()
    # ios_ver(system='iOS', release='17.2', 
    #         model='iPhone13,2', is_simulator=False)
    
    if ios_info.is_simulator:
        print("Running in simulator")
```

### Android (3.13+)
```python
import platform

if platform.system() == 'Android':
    android_info = platform.android_ver()
    # android_ver(release='14', api_level=34, 
    #             manufacturer='Google', model='Pixel 8',
    #             device='shiba', is_emulator=False)
    
    if android_info.api_level < 30:
        print("Old Android version")
```

## Production Patterns

### Pattern 1: Cross-Platform System Info Collector
```python
import platform
import sys
import os
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class SystemInfo:
    """Comprehensive system information for diagnostics."""
    
    # OS info
    os_name: str
    os_release: str
    os_version: str
    machine: str
    hostname: str
    
    # Python info
    python_version: str
    python_implementation: str
    python_compiler: str
    is_64bit: bool
    
    # Platform-specific
    platform_details: dict[str, Any]
    
    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect system information across platforms."""
        
        platform_details = {}
        os_name = platform.system()
        
        # Platform-specific details
        if os_name == 'Windows':
            release, version, csd, ptype = platform.win32_ver()
            platform_details = {
                'windows_release': release,
                'windows_version': version,
                'service_pack': csd,
                'processor_type': ptype,
                'edition': platform.win32_edition(),
                'is_iot': platform.win32_is_iot()
            }
        
        elif os_name == 'Darwin':
            release, versioninfo, machine = platform.mac_ver()
            platform_details = {
                'macos_version': release,
                'darwin_version': platform.release()
            }
        
        elif os_name == 'Linux':
            try:
                os_release = platform.freedesktop_os_release()
                platform_details = {
                    'distro_id': os_release.get('ID', ''),
                    'distro_name': os_release.get('PRETTY_NAME', ''),
                    'distro_version': os_release.get('VERSION_ID', ''),
                    'distro_like': os_release.get('ID_LIKE', '').split()
                }
            except OSError:
                lib, ver = platform.libc_ver()
                platform_details = {'libc': lib, 'libc_version': ver}
        
        elif os_name in ('iOS', 'iPadOS'):
            ios = platform.ios_ver()
            platform_details = {
                'ios_version': ios.release,
                'device_model': ios.model,
                'is_simulator': ios.is_simulator
            }
        
        elif os_name == 'Android':
            android = platform.android_ver()
            platform_details = {
                'android_version': android.release,
                'api_level': android.api_level,
                'manufacturer': android.manufacturer,
                'model': android.model,
                'is_emulator': android.is_emulator
            }
        
        return cls(
            os_name=os_name,
            os_release=platform.release(),
            os_version=platform.version(),
            machine=platform.machine(),
            hostname=platform.node(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            python_compiler=platform.python_compiler(),
            is_64bit=sys.maxsize > 2**32,
            platform_details=platform_details
        )
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.os_name} {self.os_release} ({self.machine})\n"
            f"Python {self.python_version} ({self.python_implementation})"
        )

# Usage
info = SystemInfo.collect()
print(info.summary())
print(info.to_dict())
```

### Pattern 2: Compatibility Checker
```python
import platform
import sys
from typing import NamedTuple
from enum import Enum, auto

class CompatibilityLevel(Enum):
    FULL = auto()
    PARTIAL = auto()
    UNSUPPORTED = auto()

class CompatibilityResult(NamedTuple):
    level: CompatibilityLevel
    issues: list[str]
    warnings: list[str]

def check_compatibility(
    min_python: tuple[int, int] = (3, 9),
    supported_os: list[str] = ['Linux', 'Darwin', 'Windows'],
    require_64bit: bool = False,
    min_linux_glibc: tuple[int, int] | None = None
) -> CompatibilityResult:
    """Check system compatibility with requirements."""
    
    issues = []
    warnings = []
    
    # Python version check
    py_version = sys.version_info[:2]
    if py_version < min_python:
        issues.append(
            f"Python {min_python[0]}.{min_python[1]}+ required, "
            f"found {py_version[0]}.{py_version[1]}"
        )
    
    # OS check
    os_name = platform.system()
    if os_name not in supported_os:
        issues.append(f"Unsupported OS: {os_name}")
    
    # Architecture check
    if require_64bit and sys.maxsize <= 2**32:
        issues.append("64-bit Python required")
    
    # Linux glibc check
    if os_name == 'Linux' and min_linux_glibc:
        lib, version = platform.libc_ver()
        if lib == 'glibc' and version:
            try:
                glibc_parts = tuple(map(int, version.split('.')[:2]))
                if glibc_parts < min_linux_glibc:
                    issues.append(
                        f"glibc {min_linux_glibc[0]}.{min_linux_glibc[1]}+ "
                        f"required, found {version}"
                    )
            except ValueError:
                warnings.append(f"Could not parse glibc version: {version}")
    
    # Determine compatibility level
    if issues:
        level = CompatibilityLevel.UNSUPPORTED
    elif warnings:
        level = CompatibilityLevel.PARTIAL
    else:
        level = CompatibilityLevel.FULL
    
    return CompatibilityResult(level, issues, warnings)

# Usage
result = check_compatibility(
    min_python=(3, 10),
    require_64bit=True,
    min_linux_glibc=(2, 31)
)

if result.level == CompatibilityLevel.UNSUPPORTED:
    print("Cannot run on this system:")
    for issue in result.issues:
        print(f"  - {issue}")
    sys.exit(1)
```

### Pattern 3: Platform-Specific Path Resolution
```python
import platform
import os
from pathlib import Path
from typing import Literal

def get_app_data_dir(
    app_name: str,
    scope: Literal['user', 'system'] = 'user'
) -> Path:
    """Get platform-appropriate application data directory."""
    
    os_name = platform.system()
    
    if os_name == 'Windows':
        if scope == 'user':
            base = Path(os.environ.get('LOCALAPPDATA', 
                        Path.home() / 'AppData' / 'Local'))
        else:
            base = Path(os.environ.get('PROGRAMDATA', 
                        'C:/ProgramData'))
        return base / app_name
    
    elif os_name == 'Darwin':
        if scope == 'user':
            return Path.home() / 'Library' / 'Application Support' / app_name
        else:
            return Path('/Library/Application Support') / app_name
    
    elif os_name == 'Linux':
        if scope == 'user':
            xdg_data = os.environ.get('XDG_DATA_HOME', 
                                       str(Path.home() / '.local' / 'share'))
            return Path(xdg_data) / app_name
        else:
            return Path('/var/lib') / app_name
    
    else:
        # Fallback for unknown platforms
        return Path.home() / f'.{app_name}'

def get_config_dir(app_name: str) -> Path:
    """Get platform-appropriate config directory."""
    
    os_name = platform.system()
    
    if os_name == 'Windows':
        base = Path(os.environ.get('APPDATA', 
                    Path.home() / 'AppData' / 'Roaming'))
        return base / app_name
    
    elif os_name == 'Darwin':
        return Path.home() / 'Library' / 'Preferences' / app_name
    
    else:  # Linux and others
        xdg_config = os.environ.get('XDG_CONFIG_HOME', 
                                     str(Path.home() / '.config'))
        return Path(xdg_config) / app_name

# Usage
data_dir = get_app_data_dir('myapp')
config_dir = get_config_dir('myapp')
```

### Pattern 4: Telemetry/Analytics System Info
```python
import platform
import hashlib
import sys
from typing import Any

def get_anonymous_telemetry() -> dict[str, Any]:
    """Get anonymized system info for telemetry."""
    
    # Create anonymous machine ID (stable but not identifying)
    machine_id = hashlib.sha256(
        f"{platform.node()}{platform.machine()}".encode()
    ).hexdigest()[:16]
    
    # Normalize OS name
    os_name = platform.system()
    os_family = {
        'Linux': 'linux',
        'Darwin': 'macos',
        'Windows': 'windows',
        'iOS': 'ios',
        'iPadOS': 'ios',
        'Android': 'android'
    }.get(os_name, os_name.lower())
    
    # Get OS version (major only for privacy)
    os_version = platform.release().split('.')[0] if platform.release() else ''
    
    # Architecture
    arch = platform.machine().lower()
    arch_normalized = {
        'x86_64': 'x64',
        'amd64': 'x64',
        'arm64': 'arm64',
        'aarch64': 'arm64',
        'i386': 'x86',
        'i686': 'x86'
    }.get(arch, arch)
    
    return {
        'machine_id': machine_id,
        'os': os_family,
        'os_version': os_version,
        'arch': arch_normalized,
        'python_version': '.'.join(map(str, sys.version_info[:2])),
        'python_impl': platform.python_implementation().lower(),
        'is_64bit': sys.maxsize > 2**32
    }

# Usage for analytics
telemetry = get_anonymous_telemetry()
# {'machine_id': 'a1b2c3d4e5f6g7h8', 'os': 'linux', 
#  'os_version': '5', 'arch': 'x64', ...}
```

### Pattern 5: Linux Distribution Detection
```python
import platform
from typing import NamedTuple

class LinuxDistro(NamedTuple):
    id: str           # 'ubuntu', 'fedora', 'arch'
    name: str         # 'Ubuntu 22.04.3 LTS'
    version: str      # '22.04'
    family: list[str] # ['debian'] or ['rhel', 'fedora']
    
    def is_debian_based(self) -> bool:
        return 'debian' in self.family or self.id == 'debian'
    
    def is_rhel_based(self) -> bool:
        return any(f in self.family for f in ('rhel', 'fedora', 'centos'))
    
    def is_arch_based(self) -> bool:
        return 'arch' in self.family or self.id == 'arch'

def detect_linux_distro() -> LinuxDistro | None:
    """Detect Linux distribution from os-release."""
    
    if platform.system() != 'Linux':
        return None
    
    try:
        info = platform.freedesktop_os_release()
    except OSError:
        return None
    
    distro_id = info.get('ID', 'unknown')
    name = info.get('PRETTY_NAME', info.get('NAME', distro_id))
    version = info.get('VERSION_ID', '')
    
    family = [distro_id]
    if 'ID_LIKE' in info:
        family.extend(info['ID_LIKE'].split())
    
    return LinuxDistro(
        id=distro_id,
        name=name,
        version=version,
        family=family
    )

# Usage
distro = detect_linux_distro()
if distro:
    if distro.is_debian_based():
        print("Use apt for package management")
    elif distro.is_rhel_based():
        print("Use dnf/yum for package management")
```

## Command Line Usage

```bash
# Get platform info from CLI
python -m platform
# Linux-5.15.0-91-generic-x86_64-with-glibc2.35

# Terse output
python -m platform --terse
# Linux-5.15.0-91-generic-x86_64

# Without aliases
python -m platform --nonaliased
```

## Key Insights

1. **uname() is lazy**: processor field resolved on first access (3.9+)
2. **system() vs os.uname()**: platform.system() returns user-facing name on iOS/Android
3. **freedesktop_os_release() (3.10+)**: Best way to identify Linux distros
4. **invalidate_caches() (3.14+)**: Clear cached values if hostname changes
5. **architecture() unreliable**: Use `sys.maxsize > 2**32` for 64-bit check
6. **ios_ver/android_ver (3.13+)**: Mobile platform support finally added
7. **java_ver deprecated (3.13)**: Removed in 3.15, Jython-only
