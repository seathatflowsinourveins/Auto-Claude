#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cryptography>=42.0.0",
# ]
# ///
"""
Platform Secrets Management - Encrypted Secrets Storage

Provides secure secrets management for the Ultimate Autonomous Platform.
Supports encrypted file storage, Kubernetes secrets, and environment variables.

Features:
- Fernet-based symmetric encryption (AES-128-CBC)
- Key derivation from password using PBKDF2
- Encrypted secrets file storage
- Environment variable fallback
- Kubernetes secrets integration
- Audit logging for secret access

Security:
- Master key derived from password with 480,000 PBKDF2 iterations
- Secrets encrypted at rest with Fernet
- In-memory caching with TTL
- Access audit logging

Usage:
    from secrets import SecretsManager, get_secrets_manager

    manager = get_secrets_manager(password="your-master-password")

    # Store a secret
    manager.set_secret("neo4j_password", "my-secure-password")

    # Retrieve a secret
    password = manager.get_secret("neo4j_password")
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class SecretEntry:
    """A single secret entry."""
    name: str
    encrypted_value: bytes
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecretAccessLog:
    """Audit log entry for secret access."""
    secret_name: str
    action: str  # "read", "write", "delete"
    timestamp: float
    source: str  # "file", "env", "k8s"
    success: bool


class SecretsManager:
    """
    Manages encrypted secrets for the platform.

    Supports multiple backends:
    - Encrypted file storage (default)
    - Environment variables (fallback)
    - Kubernetes secrets (production)
    """

    def __init__(
        self,
        password: Optional[str] = None,
        secrets_file: Optional[Path] = None,
        cache_ttl_seconds: float = 300.0,
        enable_audit_log: bool = True
    ):
        """
        Initialize secrets manager.

        Args:
            password: Master password for encryption (env: UAP_SECRETS_PASSWORD)
            secrets_file: Path to encrypted secrets file
            cache_ttl_seconds: How long to cache decrypted secrets
            enable_audit_log: Whether to log secret access
        """
        self._password = password or os.getenv("UAP_SECRETS_PASSWORD")
        self._secrets_file = secrets_file or Path.home() / ".uap" / "secrets.enc"
        self._cache_ttl = cache_ttl_seconds
        self._enable_audit = enable_audit_log

        # Derived encryption key
        self._fernet: Optional[Fernet] = None
        self._salt: Optional[bytes] = None

        # Caches
        self._secrets_cache: Dict[str, tuple[Any, float]] = {}  # name -> (value, expiry)
        self._audit_log: List[SecretAccessLog] = []

        # Initialize encryption if password provided
        if self._password:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize Fernet encryption with derived key."""
        if not self._password:
            return

        # Load or generate salt
        salt_file = self._secrets_file.parent / ".salt"
        if salt_file.exists():
            self._salt = salt_file.read_bytes()
        else:
            self._salt = os.urandom(16)
            salt_file.parent.mkdir(parents=True, exist_ok=True)
            salt_file.write_bytes(self._salt)

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=480000,  # OWASP 2023 recommendation
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._password.encode()))
        self._fernet = Fernet(key)

    def _log_access(
        self,
        secret_name: str,
        action: str,
        source: str,
        success: bool
    ) -> None:
        """Log secret access for auditing."""
        if not self._enable_audit:
            return

        entry = SecretAccessLog(
            secret_name=secret_name,
            action=action,
            timestamp=time.time(),
            source=source,
            success=success
        )
        self._audit_log.append(entry)

        # Keep only last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    def _encrypt(self, value: str) -> bytes:
        """Encrypt a value."""
        if not self._fernet:
            raise ValueError("Encryption not initialized - provide password")
        return self._fernet.encrypt(value.encode())

    def _decrypt(self, encrypted: bytes) -> str:
        """Decrypt a value."""
        if not self._fernet:
            raise ValueError("Encryption not initialized - provide password")
        try:
            return self._fernet.decrypt(encrypted).decode()
        except InvalidToken:
            raise ValueError("Decryption failed - invalid password or corrupted data")

    def _load_secrets_file(self) -> Dict[str, SecretEntry]:
        """Load secrets from encrypted file."""
        if not self._secrets_file.exists():
            return {}

        try:
            data = json.loads(self._secrets_file.read_text())
            return {
                name: SecretEntry(
                    name=name,
                    encrypted_value=base64.b64decode(entry["value"]),
                    created_at=entry.get("created_at", time.time()),
                    updated_at=entry.get("updated_at", time.time()),
                    metadata=entry.get("metadata", {})
                )
                for name, entry in data.items()
            }
        except Exception:
            return {}

    def _save_secrets_file(self, secrets: Dict[str, SecretEntry]) -> None:
        """Save secrets to encrypted file."""
        self._secrets_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            name: {
                "value": base64.b64encode(entry.encrypted_value).decode(),
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
                "metadata": entry.metadata
            }
            for name, entry in secrets.items()
        }
        self._secrets_file.write_text(json.dumps(data, indent=2))

    def get_secret(
        self,
        name: str,
        default: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[str]:
        """
        Get a secret value.

        Resolution order:
        1. In-memory cache (if not expired)
        2. Kubernetes secrets (if available)
        3. Encrypted file storage
        4. Environment variable (UAP_SECRET_{NAME})
        5. Default value

        Args:
            name: Secret name
            default: Default value if not found
            use_cache: Whether to use cached values

        Returns:
            Secret value or default
        """
        # Check cache first
        if use_cache and name in self._secrets_cache:
            value, expiry = self._secrets_cache[name]
            if time.time() < expiry:
                self._log_access(name, "read", "cache", True)
                return value

        # Try Kubernetes secrets
        k8s_value = self._get_k8s_secret(name)
        if k8s_value is not None:
            self._cache_secret(name, k8s_value)
            self._log_access(name, "read", "k8s", True)
            return k8s_value

        # Try encrypted file storage
        if self._fernet:
            secrets = self._load_secrets_file()
            if name in secrets:
                try:
                    value = self._decrypt(secrets[name].encrypted_value)
                    self._cache_secret(name, value)
                    self._log_access(name, "read", "file", True)
                    return value
                except ValueError:
                    self._log_access(name, "read", "file", False)

        # Try environment variable
        env_name = f"UAP_SECRET_{name.upper()}"
        env_value = os.getenv(env_name)
        if env_value is not None:
            self._cache_secret(name, env_value)
            self._log_access(name, "read", "env", True)
            return env_value

        # Return default
        self._log_access(name, "read", "default", True)
        return default

    def _get_k8s_secret(self, name: str) -> Optional[str]:
        """Get secret from Kubernetes secrets mount."""
        # Kubernetes secrets are mounted as files in /var/run/secrets/
        secret_path = Path(f"/var/run/secrets/uap/{name}")
        if secret_path.exists():
            return secret_path.read_text().strip()
        return None

    def _cache_secret(self, name: str, value: str) -> None:
        """Cache a secret value."""
        expiry = time.time() + self._cache_ttl
        self._secrets_cache[name] = (value, expiry)

    def set_secret(
        self,
        name: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a secret.

        Args:
            name: Secret name
            value: Secret value
            metadata: Optional metadata about the secret
        """
        if not self._fernet:
            raise ValueError("Cannot store secrets without encryption password")

        secrets = self._load_secrets_file()
        now = time.time()

        encrypted = self._encrypt(value)
        secrets[name] = SecretEntry(
            name=name,
            encrypted_value=encrypted,
            created_at=secrets[name].created_at if name in secrets else now,
            updated_at=now,
            metadata=metadata or {}
        )

        self._save_secrets_file(secrets)
        self._cache_secret(name, value)
        self._log_access(name, "write", "file", True)

    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.

        Args:
            name: Secret name

        Returns:
            True if deleted, False if not found
        """
        secrets = self._load_secrets_file()
        if name in secrets:
            del secrets[name]
            self._save_secrets_file(secrets)
            if name in self._secrets_cache:
                del self._secrets_cache[name]
            self._log_access(name, "delete", "file", True)
            return True

        self._log_access(name, "delete", "file", False)
        return False

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        secrets = self._load_secrets_file()
        return list(secrets.keys())

    def get_audit_log(self, limit: int = 100) -> List[SecretAccessLog]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]

    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._secrets_cache.clear()

    def rotate_key(self, new_password: str) -> None:
        """
        Rotate the encryption key.

        Re-encrypts all secrets with a new password.

        Args:
            new_password: New master password
        """
        if not self._fernet:
            raise ValueError("Cannot rotate - encryption not initialized")

        # Decrypt all secrets with old key
        secrets = self._load_secrets_file()
        decrypted = {}
        for name, entry in secrets.items():
            decrypted[name] = {
                "value": self._decrypt(entry.encrypted_value),
                "created_at": entry.created_at,
                "metadata": entry.metadata
            }

        # Initialize with new password
        self._password = new_password

        # Generate new salt
        self._salt = os.urandom(16)
        salt_file = self._secrets_file.parent / ".salt"
        salt_file.write_bytes(self._salt)

        # Re-initialize encryption
        self._init_encryption()

        # Re-encrypt all secrets
        now = time.time()
        new_secrets = {}
        for name, data in decrypted.items():
            new_secrets[name] = SecretEntry(
                name=name,
                encrypted_value=self._encrypt(data["value"]),
                created_at=data["created_at"],
                updated_at=now,
                metadata=data["metadata"]
            )

        self._save_secrets_file(new_secrets)
        self.clear_cache()


# Global instance
_global_manager: Optional[SecretsManager] = None


def get_secrets_manager(
    password: Optional[str] = None,
    **kwargs
) -> SecretsManager:
    """Get or create global secrets manager."""
    global _global_manager
    if _global_manager is None or password:
        _global_manager = SecretsManager(password=password, **kwargs)
    return _global_manager


def main():
    """Demo secrets management."""
    import tempfile

    print("=" * 60)
    print("SECRETS MANAGEMENT DEMO")
    print("=" * 60)
    print()

    # Use temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        secrets_file = Path(tmpdir) / "secrets.enc"

        # Create manager with test password
        manager = SecretsManager(
            password="demo-password-do-not-use-in-production",
            secrets_file=secrets_file
        )

        # Store secrets
        print("[>>] Storing secrets...")
        manager.set_secret("database_password", "super-secret-password")
        manager.set_secret("api_key", "sk-1234567890abcdef")
        manager.set_secret("jwt_secret", "my-jwt-signing-key")
        print("  [OK] 3 secrets stored")
        print()

        # List secrets
        print("[>>] Listing secrets...")
        for name in manager.list_secrets():
            print(f"  - {name}")
        print()

        # Retrieve secrets
        print("[>>] Retrieving secrets...")
        password = manager.get_secret("database_password")
        print(f"  database_password: {'*' * len(password) if password else 'NOT FOUND'}")

        api_key = manager.get_secret("api_key")
        print(f"  api_key: {api_key[:10]}..." if api_key else "  api_key: NOT FOUND")
        print()

        # Test environment variable fallback
        print("[>>] Testing environment variable fallback...")
        os.environ["UAP_SECRET_ENV_TEST"] = "env-value"
        env_secret = manager.get_secret("env_test")
        print(f"  env_test: {env_secret}")
        del os.environ["UAP_SECRET_ENV_TEST"]
        print()

        # Test default value
        print("[>>] Testing default value...")
        missing = manager.get_secret("missing_secret", default="default-value")
        print(f"  missing_secret: {missing}")
        print()

        # Show audit log
        print("[>>] Audit log (last 5 entries):")
        for entry in manager.get_audit_log(5):
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            print(f"  [{timestamp}] {entry.action:6} {entry.secret_name:<20} ({entry.source})")
        print()

        # Delete secret
        print("[>>] Deleting secret...")
        manager.delete_secret("jwt_secret")
        print(f"  Remaining: {manager.list_secrets()}")
        print()

        print("[OK] Secrets management demo complete")


if __name__ == "__main__":
    main()
