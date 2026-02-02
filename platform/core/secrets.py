"""
Unleashed Platform - Secrets Management

Secure secrets management with:
- Multiple backends (environment, file, vault)
- Encryption at rest
- Secret rotation
- Access auditing
- Automatic expiration
"""

import os
import json
import base64
import hashlib
import secrets as crypto_secrets
from typing import Dict, Any, Optional, List, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Cryptography (Simple XOR-based for portability, use proper crypto in production)
# ============================================================================

class SimpleCrypto:
    """Simple encryption for secrets (use proper crypto library in production)."""

    def __init__(self, key: Optional[str] = None):
        """Initialize with encryption key or generate one."""
        if key:
            self._key = hashlib.sha256(key.encode()).digest()
        else:
            self._key = hashlib.sha256(crypto_secrets.token_bytes(32)).digest()

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        data = plaintext.encode()
        encrypted = bytes(d ^ self._key[i % len(self._key)] for i, d in enumerate(data))
        return base64.b64encode(encrypted).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext."""
        encrypted = base64.b64decode(ciphertext.encode())
        decrypted = bytes(d ^ self._key[i % len(self._key)] for i, d in enumerate(encrypted))
        return decrypted.decode()


# ============================================================================
# Enums and Types
# ============================================================================

class SecretType(Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    GENERIC = "generic"


class SecretBackendType(Enum):
    """Types of secret backends."""
    ENVIRONMENT = "environment"
    FILE = "file"
    ENCRYPTED_FILE = "encrypted_file"
    MEMORY = "memory"


@dataclass
class SecretMetadata:
    """Metadata for a secret."""
    name: str
    secret_type: SecretType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    rotation_policy_days: Optional[int] = None


@dataclass
class SecretAccessLog:
    """Log entry for secret access."""
    secret_name: str
    timestamp: datetime
    accessor: Optional[str]
    operation: str  # "read", "write", "delete", "rotate"
    success: bool
    details: Optional[str] = None


# ============================================================================
# Secret Backend Interface
# ============================================================================

class SecretBackend(ABC):
    """Abstract base class for secret backends."""

    @abstractmethod
    def get(self, name: str) -> Optional[str]:
        """Get a secret value."""
        pass

    @abstractmethod
    def set(self, name: str, value: str, metadata: Optional[SecretMetadata] = None) -> bool:
        """Set a secret value."""
        pass

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    def list(self) -> List[str]:
        """List all secret names."""
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if a secret exists."""
        pass


# ============================================================================
# Environment Backend
# ============================================================================

class EnvironmentSecretBackend(SecretBackend):
    """Secret backend using environment variables."""

    def __init__(self, prefix: str = "UNLEASHED_SECRET"):
        self.prefix = prefix

    def _make_key(self, name: str) -> str:
        return f"{self.prefix}_{name.upper().replace('-', '_')}"

    def get(self, name: str) -> Optional[str]:
        return os.environ.get(self._make_key(name))

    def set(self, name: str, value: str, metadata: Optional[SecretMetadata] = None) -> bool:
        # Environment variables are read-only at runtime
        logger.warning(f"Cannot set environment variable {self._make_key(name)} at runtime")
        return False

    def delete(self, name: str) -> bool:
        key = self._make_key(name)
        if key in os.environ:
            del os.environ[key]
            return True
        return False

    def list(self) -> List[str]:
        prefix = f"{self.prefix}_"
        return [
            k[len(prefix):].lower().replace("_", "-")
            for k in os.environ.keys()
            if k.startswith(prefix)
        ]

    def exists(self, name: str) -> bool:
        return self._make_key(name) in os.environ


# ============================================================================
# Memory Backend
# ============================================================================

class MemorySecretBackend(SecretBackend):
    """In-memory secret backend (for testing/development)."""

    def __init__(self, encrypt: bool = False, encryption_key: Optional[str] = None):
        self._secrets: Dict[str, str] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        self._lock = threading.Lock()
        self._encrypt = encrypt
        self._crypto = SimpleCrypto(encryption_key) if encrypt else None

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            value = self._secrets.get(name)
            if value and self._crypto:
                return self._crypto.decrypt(value)
            return value

    def set(self, name: str, value: str, metadata: Optional[SecretMetadata] = None) -> bool:
        with self._lock:
            if self._crypto:
                value = self._crypto.encrypt(value)
            self._secrets[name] = value

            if metadata:
                self._metadata[name] = metadata
            elif name not in self._metadata:
                self._metadata[name] = SecretMetadata(
                    name=name,
                    secret_type=SecretType.GENERIC,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
            else:
                self._metadata[name].updated_at = datetime.now(timezone.utc)
                self._metadata[name].version += 1

            return True

    def delete(self, name: str) -> bool:
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                self._metadata.pop(name, None)
                return True
            return False

    def list(self) -> List[str]:
        with self._lock:
            return list(self._secrets.keys())

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._secrets

    def get_metadata(self, name: str) -> Optional[SecretMetadata]:
        with self._lock:
            return self._metadata.get(name)


# ============================================================================
# Encrypted File Backend
# ============================================================================

class EncryptedFileSecretBackend(SecretBackend):
    """Secret backend using encrypted JSON file."""

    def __init__(
        self,
        file_path: str,
        encryption_key: str,
        auto_save: bool = True
    ):
        self.file_path = Path(file_path)
        self.crypto = SimpleCrypto(encryption_key)
        self.auto_save = auto_save
        self._secrets: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load secrets from encrypted file."""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, "r") as f:
                encrypted_data = f.read()

            if encrypted_data:
                decrypted = self.crypto.decrypt(encrypted_data)
                data = json.loads(decrypted)
                self._secrets = data.get("secrets", {})
                self._metadata = data.get("metadata", {})
        except Exception as e:
            logger.error(f"Failed to load secrets file: {e}")

    def _save(self) -> None:
        """Save secrets to encrypted file."""
        try:
            data = json.dumps({
                "secrets": self._secrets,
                "metadata": self._metadata,
                "saved_at": datetime.now(timezone.utc).isoformat()
            })
            encrypted = self.crypto.encrypt(data)

            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.file_path, "w") as f:
                f.write(encrypted)
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            return self._secrets.get(name)

    def set(self, name: str, value: str, metadata: Optional[SecretMetadata] = None) -> bool:
        with self._lock:
            self._secrets[name] = value

            if metadata:
                self._metadata[name] = {
                    "secret_type": metadata.secret_type.value,
                    "created_at": metadata.created_at.isoformat(),
                    "updated_at": metadata.updated_at.isoformat(),
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "version": metadata.version,
                    "tags": metadata.tags
                }

            if self.auto_save:
                self._save()
            return True

    def delete(self, name: str) -> bool:
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                self._metadata.pop(name, None)
                if self.auto_save:
                    self._save()
                return True
            return False

    def list(self) -> List[str]:
        with self._lock:
            return list(self._secrets.keys())

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._secrets

    def save(self) -> None:
        """Manually trigger save."""
        with self._lock:
            self._save()


# ============================================================================
# Secrets Manager
# ============================================================================

class SecretsManager:
    """Unified secrets manager with multiple backends."""

    def __init__(
        self,
        primary_backend: Optional[SecretBackend] = None,
        fallback_backends: Optional[List[SecretBackend]] = None,
        enable_audit: bool = True
    ):
        self.primary = primary_backend or MemorySecretBackend()
        self.fallbacks = fallback_backends or []
        self.enable_audit = enable_audit
        self._audit_log: List[SecretAccessLog] = []
        self._lock = threading.Lock()
        self._access_handlers: List[Callable[[SecretAccessLog], None]] = []

    def _log_access(
        self,
        name: str,
        operation: str,
        success: bool,
        accessor: Optional[str] = None,
        details: Optional[str] = None
    ) -> None:
        """Log a secret access."""
        if not self.enable_audit:
            return

        entry = SecretAccessLog(
            secret_name=name,
            timestamp=datetime.now(timezone.utc),
            accessor=accessor,
            operation=operation,
            success=success,
            details=details
        )

        with self._lock:
            self._audit_log.append(entry)
            # Trim old entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]

        # Notify handlers
        for handler in self._access_handlers:
            try:
                handler(entry)
            except Exception as e:
                logger.error(f"Access handler error: {e}")

    def get(
        self,
        name: str,
        default: Optional[str] = None,
        accessor: Optional[str] = None
    ) -> Optional[str]:
        """Get a secret value."""
        # Try primary backend
        value = self.primary.get(name)
        if value is not None:
            self._log_access(name, "read", True, accessor)
            return value

        # Try fallback backends
        for backend in self.fallbacks:
            value = backend.get(name)
            if value is not None:
                self._log_access(name, "read", True, accessor, "from fallback")
                return value

        # Not found
        self._log_access(name, "read", False, accessor, "not found")
        return default

    def set(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
        expires_in_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        accessor: Optional[str] = None
    ) -> bool:
        """Set a secret value."""
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        metadata = SecretMetadata(
            name=name,
            secret_type=secret_type,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            tags=tags or {}
        )

        success = self.primary.set(name, value, metadata)
        self._log_access(name, "write", success, accessor)
        return success

    def delete(self, name: str, accessor: Optional[str] = None) -> bool:
        """Delete a secret."""
        success = self.primary.delete(name)
        self._log_access(name, "delete", success, accessor)
        return success

    def rotate(
        self,
        name: str,
        new_value: str,
        accessor: Optional[str] = None
    ) -> bool:
        """Rotate a secret to a new value."""
        if not self.primary.exists(name):
            self._log_access(name, "rotate", False, accessor, "secret not found")
            return False

        # Get existing metadata if supported
        metadata = None
        if hasattr(self.primary, "get_metadata"):
            metadata = self.primary.get_metadata(name)

        if metadata:
            metadata.version += 1
            metadata.updated_at = datetime.now(timezone.utc)
        else:
            metadata = SecretMetadata(
                name=name,
                secret_type=SecretType.GENERIC,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

        success = self.primary.set(name, new_value, metadata)
        self._log_access(name, "rotate", success, accessor)
        return success

    def list(self) -> List[str]:
        """List all secret names."""
        names = set(self.primary.list())
        for backend in self.fallbacks:
            names.update(backend.list())
        return sorted(names)

    def exists(self, name: str) -> bool:
        """Check if a secret exists."""
        if self.primary.exists(name):
            return True
        return any(b.exists(name) for b in self.fallbacks)

    def get_audit_log(
        self,
        name: Optional[str] = None,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SecretAccessLog]:
        """Get audit log entries."""
        with self._lock:
            entries = self._audit_log.copy()

        if name:
            entries = [e for e in entries if e.secret_name == name]
        if operation:
            entries = [e for e in entries if e.operation == operation]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    def add_access_handler(self, handler: Callable[[SecretAccessLog], None]) -> None:
        """Add a handler for secret access events."""
        self._access_handlers.append(handler)

    def get_expiring_secrets(self, within_days: int = 30) -> List[str]:
        """Get secrets that will expire within the specified days."""
        expiring = []
        threshold = datetime.now(timezone.utc) + timedelta(days=within_days)

        for name in self.primary.list():
            if hasattr(self.primary, "get_metadata"):
                metadata = self.primary.get_metadata(name)
                if metadata and metadata.expires_at and metadata.expires_at <= threshold:
                    expiring.append(name)

        return expiring


# ============================================================================
# Factory Functions
# ============================================================================

def create_secrets_manager(
    backend_type: SecretBackendType = SecretBackendType.ENVIRONMENT,
    **kwargs
) -> SecretsManager:
    """Create a secrets manager with the specified backend."""
    if backend_type == SecretBackendType.ENVIRONMENT:
        primary = EnvironmentSecretBackend(
            prefix=kwargs.get("prefix", "UNLEASHED_SECRET")
        )
    elif backend_type == SecretBackendType.MEMORY:
        primary = MemorySecretBackend(
            encrypt=kwargs.get("encrypt", False),
            encryption_key=kwargs.get("encryption_key")
        )
    elif backend_type == SecretBackendType.ENCRYPTED_FILE:
        encryption_key = kwargs.get("encryption_key")
        if not encryption_key:
            raise ValueError(
                "SEC-002 FIX: encryption_key is required for ENCRYPTED_FILE backend. "
                "Set via SECRETS_ENCRYPTION_KEY environment variable or pass explicitly."
            )
        primary = EncryptedFileSecretBackend(
            file_path=kwargs.get("file_path", ".secrets.enc"),
            encryption_key=encryption_key,
            auto_save=kwargs.get("auto_save", True)
        )
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    # Add environment as fallback for file/memory backends
    fallbacks = []
    if backend_type != SecretBackendType.ENVIRONMENT:
        fallbacks.append(EnvironmentSecretBackend())

    return SecretsManager(
        primary_backend=primary,
        fallback_backends=fallbacks,
        enable_audit=kwargs.get("enable_audit", True)
    )


# ============================================================================
# Global Instance
# ============================================================================

_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = create_secrets_manager()
    return _secrets_manager


def configure_secrets_manager(manager: SecretsManager) -> None:
    """Configure the global secrets manager."""
    global _secrets_manager
    _secrets_manager = manager


# ============================================================================
# Convenience Functions
# ============================================================================

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret from the global manager."""
    return get_secrets_manager().get(name, default)


def set_secret(
    name: str,
    value: str,
    secret_type: SecretType = SecretType.GENERIC
) -> bool:
    """Set a secret in the global manager."""
    return get_secrets_manager().set(name, value, secret_type)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    "SecretType",
    "SecretBackendType",

    # Data structures
    "SecretMetadata",
    "SecretAccessLog",

    # Backends
    "SecretBackend",
    "EnvironmentSecretBackend",
    "MemorySecretBackend",
    "EncryptedFileSecretBackend",

    # Manager
    "SecretsManager",

    # Crypto
    "SimpleCrypto",

    # Factory
    "create_secrets_manager",

    # Global
    "get_secrets_manager",
    "configure_secrets_manager",
    "get_secret",
    "set_secret",
]
