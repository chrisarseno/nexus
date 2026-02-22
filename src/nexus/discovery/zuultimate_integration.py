"""
Zuultimate Integration - Identity, Access, and Security Services for Nexus.

Enables Nexus to:
1. Authenticate and manage identity tokens
2. Securely store/retrieve encrypted secrets via vault
3. Enforce zero-trust authorization via ZuulGate
4. Tokenize sensitive data for external systems
5. Manage user sessions and access control
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .resource_discovery import (
    DiscoveredResource,
    ResourceDiscovery,
    ResourceSource,
    ResourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class ZuultimateConfig:
    """Configuration for Zuultimate integration."""

    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    timeout: float = 30.0
    cache_ttl: int = 300


@dataclass
class IdentityToken:
    """Opaque identity token from Zuultimate."""

    token_id: str
    created_at: datetime
    expires_at: datetime
    is_valid: bool = True


@dataclass
class TrustScore:
    """Trust/risk score for an identity."""

    score: float  # 0.0 - 1.0
    risk_level: str  # low, medium, high
    factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VaultToken:
    """Token representing encrypted data in vault."""

    token: str
    namespace: str
    created_at: datetime


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""

    allowed: bool
    permission: str
    reason: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ZuultimateIntegration:
    """
    Zuultimate integration for identity, access, and security services.

    Capabilities:
    - Identity management with opaque tokens
    - Vault operations (encrypt, decrypt, tokenize)
    - Zero-trust authorization via ZuulGate
    - Session management
    - PII reveal workflow with audit trail
    """

    def __init__(
        self,
        resource_discovery: Optional[ResourceDiscovery] = None,
        config: Optional[ZuultimateConfig] = None,
    ):
        """
        Initialize Zuultimate integration.

        Args:
            resource_discovery: Main resource discovery system
            config: Zuultimate configuration
        """
        self.resource_discovery = resource_discovery
        self.config = config or ZuultimateConfig(
            base_url=os.environ.get("ZUULTIMATE_URL", "http://localhost:8000"),
            api_key=os.environ.get("ZUULTIMATE_API_KEY"),
            access_token=os.environ.get("ZUULTIMATE_ACCESS_TOKEN"),
        )

        self._client = None
        self._identity_cache: Dict[str, IdentityToken] = {}
        self._trust_cache: Dict[str, TrustScore] = {}

        # Register with resource discovery if provided
        if resource_discovery:
            resource_discovery.register_source(ResourceSource.ZUULTIMATE, self)

        logger.info(f"ZuultimateIntegration initialized with base_url={self.config.base_url}")

    async def _get_client(self):
        """Get or create the Zuultimate SDK client."""
        if self._client is None:
            try:
                from zuultimate.sdk.client import ZuultimateClient

                self._client = ZuultimateClient(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    access_token=self.config.access_token,
                    timeout=self.config.timeout,
                )
                logger.debug("Created ZuultimateClient")
            except ImportError:
                logger.warning(
                    "Zuultimate SDK not installed. "
                    "Install with: pip install zuultimate"
                )
                raise
        return self._client

    async def discover(self) -> int:
        """
        Discover available Zuultimate services and capabilities.

        Returns:
            Number of new resources discovered
        """
        if not self.resource_discovery:
            return 0

        new_count = 0

        # Register Zuultimate services as API resources
        services = [
            {
                "id": "zuultimate-identity",
                "name": "Zuultimate Identity Service",
                "description": "Opaque identity tokens, user management, authentication, and trust scoring",
                "capabilities": ["authentication", "identity_tokens", "trust_scoring", "session_management"],
                "tags": ["identity", "security", "zero-trust"],
            },
            {
                "id": "zuultimate-vault",
                "name": "Zuultimate Vault Service",
                "description": "AES-256-GCM encryption, field-level security, tokenization, and key rotation",
                "capabilities": ["encryption", "decryption", "tokenization", "key_rotation"],
                "tags": ["encryption", "vault", "pii", "security"],
            },
            {
                "id": "zuultimate-access",
                "name": "Zuultimate ZuulGate Access Control",
                "description": "Zero-trust authorization engine with role hierarchy and context-aware policies",
                "capabilities": ["authorization", "rbac", "policy_engine", "audit_logging"],
                "tags": ["authorization", "access-control", "zero-trust", "rbac"],
            },
        ]

        for svc in services:
            resource = DiscoveredResource(
                id=svc["id"],
                name=svc["name"],
                resource_type=ResourceType.API,
                source=ResourceSource.ZUULTIMATE,
                description=svc["description"],
                url=self.config.base_url,
                documentation_url=f"{self.config.base_url}/docs",
                capabilities=svc["capabilities"],
                tags=svc["tags"],
                is_available=True,
            )
            if self.resource_discovery.register_resource(resource):
                new_count += 1

        logger.info(f"Zuultimate discovery complete: {new_count} new resources")
        return new_count

    # ==================== Identity Operations ====================

    async def authenticate(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate user and get access tokens.

        Args:
            username: User's username
            password: User's password
            mfa_code: Optional MFA code

        Returns:
            Dict with access_token, refresh_token, and user info
        """
        client = await self._get_client()

        try:
            result = await client.identity.login(
                username=username,
                password=password,
                mfa_code=mfa_code,
            )
            logger.info(f"User authenticated: {username}")
            return result
        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            raise

    async def validate_token(self, token: str) -> IdentityToken:
        """
        Validate an identity token.

        Args:
            token: Token to validate

        Returns:
            IdentityToken with validation status
        """
        # Check cache first
        if token in self._identity_cache:
            cached = self._identity_cache[token]
            if cached.expires_at > datetime.now(timezone.utc):
                return cached

        client = await self._get_client()

        try:
            result = await client.identity.validate_token(token)

            identity_token = IdentityToken(
                token_id=result.get("token_id", token[:16]),
                created_at=datetime.fromisoformat(result.get("created_at", datetime.now(timezone.utc).isoformat())),
                expires_at=datetime.fromisoformat(result.get("expires_at", datetime.now(timezone.utc).isoformat())),
                is_valid=result.get("valid", False),
            )

            # Cache valid tokens
            if identity_token.is_valid:
                self._identity_cache[token] = identity_token

            return identity_token
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return IdentityToken(
                token_id="invalid",
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc),
                is_valid=False,
            )

    async def get_trust_score(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TrustScore:
        """
        Get trust score for a user.

        Args:
            user_id: User identifier
            context: Optional context for trust calculation

        Returns:
            TrustScore with risk assessment
        """
        # Check cache
        cache_key = f"{user_id}:{hash(str(context))}"
        if cache_key in self._trust_cache:
            return self._trust_cache[cache_key]

        client = await self._get_client()

        try:
            result = await client.identity.get_trust_score(user_id, context)

            trust_score = TrustScore(
                score=result.get("score", 0.5),
                risk_level=result.get("risk_level", "medium"),
                factors=result.get("factors", {}),
            )

            self._trust_cache[cache_key] = trust_score
            return trust_score
        except Exception as e:
            logger.error(f"Failed to get trust score for {user_id}: {e}")
            return TrustScore(score=0.0, risk_level="high", factors={"error": str(e)})

    async def logout(self, session_id: Optional[str] = None) -> bool:
        """
        Logout current session or specific session.

        Args:
            session_id: Optional specific session to revoke

        Returns:
            True if successful
        """
        client = await self._get_client()

        try:
            await client.identity.logout(session_id)
            logger.info(f"Session logged out: {session_id or 'current'}")
            return True
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False

    # ==================== Vault Operations ====================

    async def encrypt_field(
        self,
        owner_id: str,
        field_name: str,
        value: str,
    ) -> str:
        """
        Encrypt a sensitive field value.

        Args:
            owner_id: Owner of the data
            field_name: Name of the field (e.g., "email", "ssn")
            value: Plaintext value to encrypt

        Returns:
            Encrypted reference identifier
        """
        client = await self._get_client()

        try:
            result = await client.vault.encrypt(
                owner_id=owner_id,
                field_name=field_name,
                value=value,
            )
            logger.debug(f"Encrypted field {field_name} for owner {owner_id}")
            return result.get("reference_id", "")
        except Exception as e:
            logger.error(f"Encryption failed for {field_name}: {e}")
            raise

    async def decrypt_field(
        self,
        owner_id: str,
        field_name: str,
        reveal_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Decrypt a sensitive field value.

        Requires authorization and creates audit log entry.

        Args:
            owner_id: Owner of the data
            field_name: Name of the field to decrypt
            reveal_token: Optional reveal token for PII access

        Returns:
            Decrypted plaintext value
        """
        client = await self._get_client()

        try:
            result = await client.vault.decrypt(
                owner_id=owner_id,
                field_name=field_name,
                reveal_token=reveal_token,
            )
            logger.info(f"Decrypted field {field_name} for owner {owner_id}")
            return result.get("value")
        except Exception as e:
            logger.error(f"Decryption failed for {field_name}: {e}")
            raise

    async def tokenize(
        self,
        value: str,
        namespace: str = "default",
        allow_duplicate: bool = False,
    ) -> VaultToken:
        """
        Tokenize a sensitive value for external systems.

        Args:
            value: Value to tokenize
            namespace: Namespace for isolation
            allow_duplicate: Whether to allow duplicate tokens

        Returns:
            VaultToken with opaque token reference
        """
        client = await self._get_client()

        try:
            result = await client.vault.tokenize(
                value=value,
                namespace=namespace,
                allow_duplicate=allow_duplicate,
            )

            vault_token = VaultToken(
                token=result.get("token", ""),
                namespace=namespace,
                created_at=datetime.now(timezone.utc),
            )

            logger.debug(f"Tokenized value in namespace {namespace}")
            return vault_token
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

    async def detokenize(
        self,
        token: str,
        reveal_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Detokenize a vault token to get original value.

        Requires authorization.

        Args:
            token: Vault token to detokenize
            reveal_token: Optional reveal token for access

        Returns:
            Original plaintext value
        """
        client = await self._get_client()

        try:
            result = await client.vault.detokenize(
                token=token,
                reveal_token=reveal_token,
            )
            logger.info("Detokenized value")
            return result.get("value")
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            raise

    # ==================== Authorization Operations ====================

    async def check_permission(
        self,
        permission: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationResult:
        """
        Check if current user has a specific permission.

        Args:
            permission: Permission to check (e.g., "PII_REVEAL", "VAULT_WRITE")
            context: Optional context for policy evaluation

        Returns:
            AuthorizationResult with decision
        """
        client = await self._get_client()

        try:
            result = await client.access.check_permission(
                permission=permission,
                context=context,
            )

            return AuthorizationResult(
                allowed=result.get("allowed", False),
                permission=permission,
                reason=result.get("reason"),
                context=context or {},
            )
        except Exception as e:
            logger.error(f"Permission check failed for {permission}: {e}")
            return AuthorizationResult(
                allowed=False,
                permission=permission,
                reason=str(e),
            )

    async def authorize(
        self,
        permission: str,
        context: Optional[Dict[str, Any]] = None,
        raise_on_deny: bool = True,
    ) -> bool:
        """
        Authorize an operation with optional exception on denial.

        Args:
            permission: Permission required
            context: Context for policy evaluation
            raise_on_deny: Whether to raise exception on denial

        Returns:
            True if authorized

        Raises:
            PermissionError: If denied and raise_on_deny is True
        """
        result = await self.check_permission(permission, context)

        if not result.allowed:
            if raise_on_deny:
                raise PermissionError(
                    f"Access denied for permission '{permission}': {result.reason}"
                )
            return False

        return True

    async def get_user_permissions(self) -> List[str]:
        """
        Get all permissions for current user.

        Returns:
            List of permission strings
        """
        client = await self._get_client()

        try:
            result = await client.access.get_permissions()
            return result.get("permissions", [])
        except Exception as e:
            logger.error(f"Failed to get permissions: {e}")
            return []

    async def get_user_roles(self) -> List[Dict[str, Any]]:
        """
        Get all roles assigned to current user.

        Returns:
            List of role objects
        """
        client = await self._get_client()

        try:
            result = await client.access.get_roles()
            return result.get("roles", [])
        except Exception as e:
            logger.error(f"Failed to get roles: {e}")
            return []

    # ==================== Utility Methods ====================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Zuultimate service health.

        Returns:
            Health status dict
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "service": "zuultimate",
                            "details": data,
                        }
                    return {
                        "status": "unhealthy",
                        "service": "zuultimate",
                        "error": f"HTTP {response.status}",
                    }
        except Exception as e:
            return {
                "status": "unavailable",
                "service": "zuultimate",
                "error": str(e),
            }

    def clear_cache(self) -> None:
        """Clear all cached identity and trust data."""
        self._identity_cache.clear()
        self._trust_cache.clear()
        logger.debug("Cleared Zuultimate caches")

    async def close(self) -> None:
        """Close the integration and cleanup resources."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Zuultimate client: {e}")
            self._client = None

        self.clear_cache()
        logger.info("ZuultimateIntegration closed")
