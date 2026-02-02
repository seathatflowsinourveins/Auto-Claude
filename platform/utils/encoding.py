# -*- coding: utf-8 -*-
"""
Unicode Encoding Utilities for Cross-Platform Compatibility

This module provides robust UTF-8 encoding handling for Windows and Unix systems.
Based on PEP 528 (Windows Console UTF-8) and PEP 686 (UTF-8 Mode Default).

Usage:
    from utils.encoding import configure_encoding, safe_print, safe_str

    # Call once at application startup
    configure_encoding()

    # Use safe_print for Unicode output
    safe_print("Results: 3 items [OK]")

    # Convert strings safely
    text = safe_str(some_object)
"""

import os
import sys
import io
import logging
from typing import Any, Optional, Dict, TextIO
from functools import wraps

logger = logging.getLogger(__name__)

# ASCII-safe replacements for common Unicode symbols
UNICODE_REPLACEMENTS: Dict[str, str] = {
    "\u2713": "[OK]",      # checkmark
    "\u2714": "[OK]",      # heavy checkmark
    "\u2715": "[X]",       # X mark
    "\u2716": "[X]",       # heavy X mark
    "\u2717": "[X]",       # ballot X
    "\u2718": "[X]",       # heavy ballot X
    "\u2022": "*",         # bullet
    "\u2023": ">",         # triangular bullet
    "\u25CF": "*",         # black circle
    "\u25CB": "o",         # white circle
    "\u2192": "->",        # right arrow
    "\u2190": "<-",        # left arrow
    "\u2194": "<->",       # left right arrow
    "\u21D2": "=>",        # double right arrow
    "\u2026": "...",       # ellipsis
    "\u2018": "'",         # left single quote
    "\u2019": "'",         # right single quote
    "\u201C": '"',         # left double quote
    "\u201D": '"',         # right double quote
    "\u2014": "--",        # em dash
    "\u2013": "-",         # en dash
    "\u00A0": " ",         # non-breaking space
    "\u00B7": "*",         # middle dot
    "\u2605": "*",         # star
    "\u2606": "*",         # white star
    "\u2764": "<3",        # heart
    "\u00A9": "(c)",       # copyright
    "\u00AE": "(R)",       # registered
    "\u2122": "(TM)",      # trademark
    "\u00B0": "deg",       # degree
    "\u00B1": "+/-",       # plus minus
    "\u00D7": "x",         # multiplication
    "\u00F7": "/",         # division
    "\u221A": "sqrt",      # square root
    "\u221E": "inf",       # infinity
    "\u2248": "~=",        # approximately equal
    "\u2260": "!=",        # not equal
    "\u2264": "<=",        # less than or equal
    "\u2265": ">=",        # greater than or equal
    "\u03B1": "alpha",     # greek alpha
    "\u03B2": "beta",      # greek beta
    "\u03B3": "gamma",     # greek gamma
    "\u03C0": "pi",        # greek pi
}

# Status indicators (ASCII-safe)
STATUS = {
    "ok": "[OK]",
    "fail": "[FAIL]",
    "warn": "[WARN]",
    "info": "[INFO]",
    "skip": "[SKIP]",
    "done": "[DONE]",
    "error": "[ERROR]",
    "check": "[/]",
    "cross": "[X]",
    "arrow": "->",
    "bullet": "*",
    "star": "*",
}


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32" or os.name == "nt"


def get_encoding() -> str:
    """Get the current stdout encoding."""
    if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
        return sys.stdout.encoding.lower()
    return "utf-8"


def supports_unicode() -> bool:
    """Check if the current environment supports full Unicode output."""
    encoding = get_encoding()
    return encoding in ("utf-8", "utf8", "utf_8")


def configure_encoding(force_utf8: bool = True) -> bool:
    """
    Configure UTF-8 encoding for stdin, stdout, and stderr.

    This should be called once at application startup before any I/O.

    Args:
        force_utf8: If True, attempt to reconfigure streams to UTF-8

    Returns:
        True if UTF-8 encoding was successfully configured
    """
    configured = False

    # Set environment variables for child processes
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Try to reconfigure streams (Python 3.7+)
    if force_utf8 and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            if sys.stdin and hasattr(sys.stdin, "reconfigure"):
                sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            configured = True
            logger.debug("Configured UTF-8 encoding via reconfigure()")
        except Exception as e:
            logger.debug(f"Could not reconfigure streams: {e}")

    # Fallback: wrap streams with TextIOWrapper
    if not configured and force_utf8:
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True
                )
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True
                )
            configured = True
            logger.debug("Configured UTF-8 encoding via TextIOWrapper")
        except Exception as e:
            logger.debug(f"Could not wrap streams: {e}")

    # On Windows, try to set console mode
    if is_windows():
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Set console output code page to UTF-8 (65001)
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
            logger.debug("Set Windows console to UTF-8 code page")
        except Exception as e:
            logger.debug(f"Could not set Windows console code page: {e}")

    return configured or supports_unicode()


def safe_str(obj: Any, fallback: str = "?") -> str:
    """
    Convert any object to a string that's safe for ASCII output.

    Replaces Unicode characters with ASCII equivalents.

    Args:
        obj: Object to convert
        fallback: Character to use for unreplaceable Unicode

    Returns:
        ASCII-safe string representation
    """
    try:
        text = str(obj)
    except Exception:
        return fallback * 3

    # Apply Unicode replacements
    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)

    # Encode to ASCII, replacing unknown characters
    try:
        text = text.encode("ascii", errors="replace").decode("ascii")
    except Exception:
        # Final fallback: remove non-ASCII
        text = "".join(c if ord(c) < 128 else fallback for c in text)

    return text


def safe_print(
    *args,
    sep: str = " ",
    end: str = "\n",
    file: Optional[TextIO] = None,
    flush: bool = False,
    ascii_safe: bool = None
) -> None:
    """
    Print function that handles Unicode safely on all platforms.

    On Windows with non-UTF8 console, automatically converts to ASCII-safe output.

    Args:
        *args: Objects to print
        sep: Separator between objects
        end: String to append at end
        file: Output stream (default: sys.stdout)
        flush: Whether to flush the stream
        ascii_safe: Force ASCII-safe output (auto-detected if None)
    """
    if file is None:
        file = sys.stdout

    # Auto-detect if we need ASCII-safe output
    if ascii_safe is None:
        ascii_safe = not supports_unicode() and is_windows()

    # Convert arguments to strings
    if ascii_safe:
        str_args = [safe_str(arg) for arg in args]
    else:
        str_args = [str(arg) for arg in args]

    # Join and print
    output = sep.join(str_args)

    try:
        print(output, end=end, file=file, flush=flush)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe
        output = safe_str(output)
        print(output, end=end, file=file, flush=flush)


def make_safe_formatter() -> logging.Formatter:
    """
    Create a logging formatter that handles Unicode safely.

    Returns:
        Configured logging.Formatter
    """
    class SafeFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            result = super().format(record)
            if is_windows() and not supports_unicode():
                return safe_str(result)
            return result

    return SafeFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def configure_safe_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging with Unicode-safe output.

    Args:
        level: Logging level
        format_string: Custom format string (optional)
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(make_safe_formatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)


def status(name: str) -> str:
    """
    Get an ASCII-safe status indicator.

    Args:
        name: Status name (ok, fail, warn, info, etc.)

    Returns:
        ASCII-safe status string
    """
    return STATUS.get(name.lower(), f"[{name.upper()}]")


# Convenience functions using STATUS
def ok(msg: str = "") -> str:
    """Return OK status with optional message."""
    return f"{STATUS['ok']} {msg}".strip()


def fail(msg: str = "") -> str:
    """Return FAIL status with optional message."""
    return f"{STATUS['fail']} {msg}".strip()


def warn(msg: str = "") -> str:
    """Return WARN status with optional message."""
    return f"{STATUS['warn']} {msg}".strip()


def info(msg: str = "") -> str:
    """Return INFO status with optional message."""
    return f"{STATUS['info']} {msg}".strip()


# Auto-configure on import if this is the main module
if __name__ == "__main__":
    configure_encoding()

    # Test output
    print("Encoding Test Results:")
    print(f"  Platform: {sys.platform}")
    print(f"  Encoding: {get_encoding()}")
    print(f"  Unicode support: {supports_unicode()}")
    print()
    safe_print(f"  {ok('Configuration successful')}")
    safe_print(f"  {info('Unicode handling enabled')}")
