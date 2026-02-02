#!/usr/bin/env python3
"""
Pydantic Compatibility Layer for Python 3.14+
Part of V34 Architecture - Phase 10 Fix.

This module provides a unified interface that works with both Pydantic v1 and v2,
handling the breaking changes in ForwardRef._evaluate and other Python 3.14+
incompatibilities.

The key issue: Pydantic V1's typing utilities use ForwardRef._evaluate() which
was deprecated in Python 3.14 and will be removed in 3.16. This shim provides
compatible alternatives.

Usage:
    from core.compat import (
        model_to_dict,
        model_to_json,
        model_validate,
        model_schema,
        PYDANTIC_V2,
    )
"""

from __future__ import annotations

import sys
import json
import warnings
from typing import Any, Dict, Type, TypeVar, Optional, Union, get_type_hints
from dataclasses import dataclass, field

# =============================================================================
# VERSION DETECTION
# =============================================================================

PYTHON_VERSION = sys.version_info[:2]
PYTHON_314_PLUS = PYTHON_VERSION >= (3, 14)

# Detect Pydantic version
try:
    import pydantic
    PYDANTIC_VERSION = tuple(int(x) for x in pydantic.VERSION.split(".")[:2])
    PYDANTIC_V2 = PYDANTIC_VERSION >= (2, 0)
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_VERSION = (0, 0)
    PYDANTIC_V2 = False
    PYDANTIC_AVAILABLE = False

# =============================================================================
# COMPATIBILITY EXCEPTIONS
# =============================================================================

class CompatibilityError(Exception):
    """Raised when compatibility layer cannot handle a specific case."""
    pass


class PydanticCompatError(CompatibilityError):
    """Raised when Pydantic compatibility fails."""

    def __init__(self, operation: str, original_error: Exception):
        self.operation = operation
        self.original_error = original_error
        msg = f"""
============================================================
PYDANTIC COMPATIBILITY ERROR: {operation}
============================================================

Python Version: {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}
Pydantic Version: {'.'.join(str(x) for x in PYDANTIC_VERSION)}
Pydantic V2: {PYDANTIC_V2}

Original Error:
  {type(original_error).__name__}: {original_error}

This error occurs because the SDK uses Pydantic V1 patterns
that are incompatible with Python 3.14+.

Workaround: Use the compatibility functions from core.compat
============================================================
"""
        super().__init__(msg)


# =============================================================================
# TYPE VARIABLE FOR GENERIC MODELS
# =============================================================================

T = TypeVar("T")


# =============================================================================
# FORWARD REFERENCE RESOLVER (Python 3.14+ Compatible)
# =============================================================================

def resolve_forward_ref(ref: Any, globalns: Optional[Dict] = None,
                         localns: Optional[Dict] = None) -> Any:
    """
    Resolve a ForwardRef in a Python 3.14+ compatible way.

    In Python 3.14+, ForwardRef._evaluate is deprecated. This function
    uses the new evaluate_forward_ref() API when available.
    """
    from typing import ForwardRef

    if not isinstance(ref, ForwardRef):
        return ref

    globalns = globalns or {}
    localns = localns or {}

    if PYTHON_314_PLUS:
        # Python 3.14+ uses the new API
        try:
            # First try the new typing.evaluate_forward_ref
            from typing import evaluate_forward_ref  # type: ignore[attr-defined]
            return evaluate_forward_ref(ref, globals=globalns, locals=localns)  # type: ignore
        except (ImportError, TypeError):
            # Fallback to ForwardRef.evaluate() method
            try:
                return ref.evaluate(globals=globalns, locals=localns)  # type: ignore
            except (AttributeError, TypeError):
                # Last resort: suppress warning and use deprecated API
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    return ref._evaluate(globalns, localns,  # type: ignore
                                         type_params=(), recursive_guard=frozenset())
    else:
        # Python < 3.14 uses the old API
        return ref._evaluate(globalns, localns, frozenset())  # type: ignore


# =============================================================================
# MODEL UTILITIES (V1/V2 Compatible)
# =============================================================================

def model_to_dict(model: Any,
                   exclude_none: bool = False,
                   exclude_unset: bool = False,
                   by_alias: bool = False) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a dictionary.

    Works with both Pydantic V1 and V2 models.

    Args:
        model: Pydantic model instance
        exclude_none: Exclude None values
        exclude_unset: Exclude unset values (V2 only)
        by_alias: Use field aliases

    Returns:
        Dictionary representation of the model
    """
    if not PYDANTIC_AVAILABLE:
        # Fallback for non-Pydantic objects
        if hasattr(model, "__dict__"):
            result = dict(model.__dict__)
            if exclude_none:
                result = {k: v for k, v in result.items() if v is not None}
            return result
        raise PydanticCompatError("model_to_dict",
                                   ValueError("Object has no __dict__"))

    try:
        if PYDANTIC_V2:
            # Pydantic V2 API
            return model.model_dump(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                by_alias=by_alias
            )
        else:
            # Pydantic V1 API
            return model.dict(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                by_alias=by_alias
            )
    except Exception as e:
        raise PydanticCompatError("model_to_dict", e)


def model_to_json(model: Any,
                   indent: Optional[int] = None,
                   exclude_none: bool = False,
                   by_alias: bool = False) -> str:
    """
    Convert a Pydantic model to a JSON string.

    Works with both Pydantic V1 and V2 models.

    Args:
        model: Pydantic model instance
        indent: JSON indentation level
        exclude_none: Exclude None values
        by_alias: Use field aliases

    Returns:
        JSON string representation
    """
    if not PYDANTIC_AVAILABLE:
        data = model_to_dict(model, exclude_none=exclude_none)
        return json.dumps(data, indent=indent, default=str)

    try:
        if PYDANTIC_V2:
            # Pydantic V2 API
            return model.model_dump_json(
                indent=indent,
                exclude_none=exclude_none,
                by_alias=by_alias
            )
        else:
            # Pydantic V1 API
            return model.json(
                indent=indent,
                exclude_none=exclude_none,
                by_alias=by_alias
            )
    except Exception as e:
        raise PydanticCompatError("model_to_json", e)


def model_validate(model_class: Type[T], data: Union[Dict, Any]) -> T:
    """
    Validate data and create a model instance.

    Works with both Pydantic V1 and V2 models.

    Args:
        model_class: The Pydantic model class
        data: Dictionary or object to validate

    Returns:
        Validated model instance
    """
    if not PYDANTIC_AVAILABLE:
        raise PydanticCompatError("model_validate",
                                   ImportError("Pydantic not available"))

    try:
        if PYDANTIC_V2:
            # Pydantic V2 API
            return model_class.model_validate(data)
        else:
            # Pydantic V1 API
            if isinstance(data, dict):
                return model_class(**data)
            else:
                return model_class.parse_obj(data)
    except Exception as e:
        raise PydanticCompatError("model_validate", e)


def model_schema(model_class: Type[Any],
                  by_alias: bool = True,
                  ref_template: str = "{model}") -> Dict[str, Any]:
    """
    Get the JSON schema for a Pydantic model.

    Works with both Pydantic V1 and V2 models.

    Args:
        model_class: The Pydantic model class
        by_alias: Use field aliases in schema
        ref_template: Template for $ref values

    Returns:
        JSON schema dictionary
    """
    if not PYDANTIC_AVAILABLE:
        raise PydanticCompatError("model_schema",
                                   ImportError("Pydantic not available"))

    try:
        if PYDANTIC_V2:
            # Pydantic V2 API
            return model_class.model_json_schema(
                by_alias=by_alias,
                ref_template=ref_template
            )
        else:
            # Pydantic V1 API
            return model_class.schema(by_alias=by_alias)
    except Exception as e:
        raise PydanticCompatError("model_schema", e)


def model_fields(model_class: Type[Any]) -> Dict[str, Any]:
    """
    Get the fields of a Pydantic model.

    Works with both Pydantic V1 and V2 models.

    Args:
        model_class: The Pydantic model class

    Returns:
        Dictionary of field name to field info
    """
    if not PYDANTIC_AVAILABLE:
        raise PydanticCompatError("model_fields",
                                   ImportError("Pydantic not available"))

    try:
        if PYDANTIC_V2:
            # Pydantic V2 API
            return dict(model_class.model_fields)
        else:
            # Pydantic V1 API
            return dict(model_class.__fields__)
    except Exception as e:
        raise PydanticCompatError("model_fields", e)


# =============================================================================
# TYPE HINT UTILITIES (Python 3.14+ Compatible)
# =============================================================================

def get_safe_type_hints(obj: Any,
                         globalns: Optional[Dict] = None,
                         localns: Optional[Dict] = None,
                         include_extras: bool = False) -> Dict[str, Any]:
    """
    Get type hints with Python 3.14+ compatibility.

    This wraps typing.get_type_hints to handle ForwardRef resolution
    properly on Python 3.14+.

    Args:
        obj: Object to get type hints for
        globalns: Global namespace for resolution
        localns: Local namespace for resolution
        include_extras: Include Annotated extras

    Returns:
        Dictionary of parameter name to type
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return get_type_hints(obj, globalns=globalns, localns=localns,
                                   include_extras=include_extras)
    except Exception as e:
        # Fallback: return __annotations__ if available
        if hasattr(obj, "__annotations__"):
            return dict(obj.__annotations__)
        return {}


# =============================================================================
# DATACLASS COMPATIBILITY
# =============================================================================

@dataclass
class CompatInfo:
    """Information about the current compatibility status."""
    python_version: tuple = field(default_factory=lambda: PYTHON_VERSION)
    pydantic_version: tuple = field(default_factory=lambda: PYDANTIC_VERSION)
    pydantic_v2: bool = PYDANTIC_V2
    pydantic_available: bool = PYDANTIC_AVAILABLE
    python_314_plus: bool = PYTHON_314_PLUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": f"{self.python_version[0]}.{self.python_version[1]}",
            "pydantic_version": f"{self.pydantic_version[0]}.{self.pydantic_version[1]}",
            "pydantic_v2": self.pydantic_v2,
            "pydantic_available": self.pydantic_available,
            "python_314_plus": self.python_314_plus,
        }


def get_compat_info() -> CompatInfo:
    """Get current compatibility information."""
    return CompatInfo()


# =============================================================================
# PYDANTIC SETTINGS COMPATIBILITY
# =============================================================================

def create_settings_class(
    name: str,
    fields: Dict[str, tuple],  # {field_name: (type, default)}
    env_prefix: str = "",
    env_file: Optional[str] = None
) -> Type:
    """
    Create a Pydantic Settings class compatible with both V1 and V2.

    Args:
        name: Class name
        fields: Dictionary of {field_name: (type, default_value)}
        env_prefix: Environment variable prefix
        env_file: Path to .env file

    Returns:
        Settings class
    """
    # Capture parameters for use in nested class definitions
    _env_prefix = env_prefix
    _env_file = env_file

    if not PYDANTIC_AVAILABLE:
        # Fallback: create a simple dataclass
        from dataclasses import make_dataclass
        field_list = [(k, v[0], field(default=v[1]))
                      for k, v in fields.items()]
        return make_dataclass(name, field_list)

    try:
        if PYDANTIC_V2:
            from pydantic_settings import BaseSettings  # type: ignore[import-untyped]
            from pydantic import Field

            # Build field definitions
            namespace: Dict[str, Any] = {"__annotations__": {}}
            for field_name, (field_type, default) in fields.items():
                namespace["__annotations__"][field_name] = field_type
                namespace[field_name] = Field(default=default)

            # Pydantic V2 uses model_config dict
            namespace["model_config"] = {"env_prefix": _env_prefix,
                                          "env_file": _env_file}

            return type(name, (BaseSettings,), namespace)  # type: ignore[return-value]
        else:
            from pydantic import BaseSettings, Field  # type: ignore[attr-defined, no-redef]

            namespace: Dict[str, Any] = {"__annotations__": {}}  # type: ignore[no-redef]
            for field_name, (field_type, default) in fields.items():
                namespace["__annotations__"][field_name] = field_type
                namespace[field_name] = Field(default=default)

            # Pydantic V1 uses Config class
            config_attrs = {"env_prefix": _env_prefix, "env_file": _env_file}
            ConfigClass = type("Config", (), config_attrs)
            namespace["Config"] = ConfigClass

            return type(name, (BaseSettings,), namespace)  # type: ignore[return-value]
    except ImportError:
        # pydantic-settings not installed, use base pydantic
        from pydantic import BaseModel, Field

        namespace: Dict[str, Any] = {"__annotations__": {}}  # type: ignore[no-redef]
        for field_name, (field_type, default) in fields.items():
            namespace["__annotations__"][field_name] = field_type
            namespace[field_name] = Field(default=default)

        return type(name, (BaseModel,), namespace)  # type: ignore[return-value]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version info
    "PYTHON_VERSION",
    "PYTHON_314_PLUS",
    "PYDANTIC_VERSION",
    "PYDANTIC_V2",
    "PYDANTIC_AVAILABLE",
    # Exceptions
    "CompatibilityError",
    "PydanticCompatError",
    # Model utilities
    "model_to_dict",
    "model_to_json",
    "model_validate",
    "model_schema",
    "model_fields",
    # Type utilities
    "resolve_forward_ref",
    "get_safe_type_hints",
    # Info
    "CompatInfo",
    "get_compat_info",
    # Settings
    "create_settings_class",
]
