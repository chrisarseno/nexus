"""
Vinzy-Engine Integration - License Management for Nexus.

Enables Nexus to:
1. Validate and manage software licenses
2. Track entitlements and feature access
3. Manage machine activations
4. Record and monitor usage metrics
5. Handle offline license validation
"""

import asyncio
import logging
import os
import platform
import socket
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
class VinzyConfig:
    """Configuration for Vinzy-Engine integration."""

    server_url: str = "http://localhost:8080"
    license_key: Optional[str] = None
    api_key: Optional[str] = None
    product_id: Optional[str] = None
    cache_ttl: int = 300
    timeout: float = 30.0


@dataclass
class License:
    """License information from Vinzy."""

    id: str
    key: str
    status: str  # active, expired, suspended, revoked
    product_code: str
    customer_id: str
    tier: str
    machines_limit: Optional[int] = None
    machines_used: int = 0
    expires_at: Optional[datetime] = None
    features: List[str] = field(default_factory=list)
    entitlements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entitlement:
    """Feature entitlement details."""

    feature: str
    enabled: bool
    limit: Optional[int] = None
    used: int = 0
    remaining: Optional[int] = None

    @property
    def is_available(self) -> bool:
        """Check if entitlement is available for use."""
        if not self.enabled:
            return False
        if self.limit is None:
            return True
        return self.remaining is None or self.remaining > 0


@dataclass
class ValidationResult:
    """Result of license validation."""

    valid: bool
    license: Optional[License] = None
    code: str = ""  # OK, EXPIRED, SUSPENDED, REVOKED, NOT_FOUND, etc.
    message: str = ""
    features: List[str] = field(default_factory=list)
    entitlements: List[Entitlement] = field(default_factory=list)


@dataclass
class ActivationResult:
    """Result of machine activation."""

    success: bool
    machine_id: Optional[str] = None
    license: Optional[License] = None
    code: str = ""
    message: str = ""


@dataclass
class UsageResult:
    """Result of usage recording."""

    success: bool
    metric: str
    value_added: float = 0.0
    total_value: float = 0.0
    limit: Optional[float] = None
    remaining: Optional[float] = None
    code: str = ""


class VinzyIntegration:
    """
    Vinzy-Engine integration for license management.

    Capabilities:
    - License validation and caching
    - Entitlement checking for feature gating
    - Machine activation and deactivation
    - Usage tracking and metering
    - Offline validation support
    """

    def __init__(
        self,
        resource_discovery: Optional[ResourceDiscovery] = None,
        config: Optional[VinzyConfig] = None,
    ):
        """
        Initialize Vinzy integration.

        Args:
            resource_discovery: Main resource discovery system
            config: Vinzy configuration
        """
        self.resource_discovery = resource_discovery
        self.config = config or VinzyConfig(
            server_url=os.environ.get("VINZY_SERVER", "http://localhost:8080"),
            license_key=os.environ.get("VINZY_LICENSE_KEY"),
            api_key=os.environ.get("VINZY_API_KEY"),
            product_id=os.environ.get("VINZY_PRODUCT_ID"),
        )

        self._client = None
        self._cached_license: Optional[License] = None
        self._cache_timestamp: Optional[datetime] = None
        self._entitlement_cache: Dict[str, Entitlement] = {}
        self._machine_fingerprint: Optional[str] = None

        # Register with resource discovery if provided
        if resource_discovery:
            resource_discovery.register_source(ResourceSource.VINZY, self)

        logger.info(f"VinzyIntegration initialized with server_url={self.config.server_url}")

    async def _get_client(self):
        """Get or create the Vinzy SDK client."""
        if self._client is None:
            try:
                from vinzy_engine import LicenseClient

                self._client = LicenseClient(
                    server_url=self.config.server_url,
                    license_key=self.config.license_key,
                    cache_ttl=self.config.cache_ttl,
                    timeout=int(self.config.timeout),
                )
                logger.debug("Created Vinzy LicenseClient")
            except ImportError:
                logger.warning(
                    "Vinzy-Engine SDK not installed. "
                    "Install with: pip install vinzy-engine"
                )
                raise
        return self._client

    def _get_fingerprint(self) -> str:
        """Generate or return cached machine fingerprint."""
        if self._machine_fingerprint:
            return self._machine_fingerprint

        import hashlib

        # Collect machine identifiers
        identifiers = [
            platform.node(),  # hostname
            platform.machine(),  # machine type
            platform.processor(),  # processor
            str(os.getpid()),  # process ID (for testing)
        ]

        # Try to get MAC address
        try:
            import uuid
            mac = uuid.getnode()
            identifiers.append(str(mac))
        except Exception:
            pass

        # Create fingerprint hash
        combined = "|".join(identifiers)
        self._machine_fingerprint = hashlib.sha256(combined.encode()).hexdigest()[:32]

        return self._machine_fingerprint

    async def discover(self) -> int:
        """
        Discover Vinzy license services.

        Returns:
            Number of new resources discovered
        """
        if not self.resource_discovery:
            return 0

        new_count = 0

        # Register Vinzy service as API resource
        resource = DiscoveredResource(
            id="vinzy-license-manager",
            name="Vinzy License Manager",
            resource_type=ResourceType.API,
            source=ResourceSource.VINZY,
            description="Software license validation, entitlements, machine activation, and usage tracking",
            url=self.config.server_url,
            documentation_url=f"{self.config.server_url}/docs",
            capabilities=[
                "license_validation",
                "entitlement_checking",
                "machine_activation",
                "usage_tracking",
                "offline_validation",
            ],
            tags=["licensing", "entitlements", "metering", "activation"],
            is_available=True,
        )

        if self.resource_discovery.register_resource(resource):
            new_count += 1

        logger.info(f"Vinzy discovery complete: {new_count} new resources")
        return new_count

    # ==================== License Validation ====================

    async def validate(
        self,
        license_key: Optional[str] = None,
        fingerprint: Optional[str] = None,
        force_refresh: bool = False,
    ) -> ValidationResult:
        """
        Validate the license.

        Args:
            license_key: License key to validate (uses configured key if not provided)
            fingerprint: Machine fingerprint (auto-generated if not provided)
            force_refresh: Force server validation, ignore cache

        Returns:
            ValidationResult with license status
        """
        # Check cache unless force refresh
        if not force_refresh and self._cached_license and self._cache_timestamp:
            cache_age = (datetime.now(timezone.utc) - self._cache_timestamp).total_seconds()
            if cache_age < self.config.cache_ttl:
                return ValidationResult(
                    valid=self._cached_license.status == "active",
                    license=self._cached_license,
                    code="OK" if self._cached_license.status == "active" else self._cached_license.status.upper(),
                    message="Cached validation",
                    features=self._cached_license.features,
                    entitlements=list(self._entitlement_cache.values()),
                )

        client = await self._get_client()

        try:
            result = client.validate(fingerprint=fingerprint or self._get_fingerprint())

            if result.valid and result.license:
                license_obj = License(
                    id=result.license.id,
                    key=result.license.key,
                    status=result.license.status,
                    product_code=result.license.product_code,
                    customer_id=result.license.customer_id,
                    tier=result.license.tier,
                    machines_limit=getattr(result.license, "machines_limit", None),
                    machines_used=getattr(result.license, "machines_used", 0),
                    expires_at=result.license.expires_at,
                    features=result.license.features or [],
                    entitlements=result.license.entitlements or {},
                )

                # Cache the license
                self._cached_license = license_obj
                self._cache_timestamp = datetime.now(timezone.utc)

                # Cache entitlements
                for ent in result.entitlements or []:
                    self._entitlement_cache[ent.feature] = Entitlement(
                        feature=ent.feature,
                        enabled=ent.enabled,
                        limit=ent.limit,
                        used=ent.used,
                        remaining=ent.remaining,
                    )

                return ValidationResult(
                    valid=True,
                    license=license_obj,
                    code=result.code,
                    message=result.message,
                    features=result.features or [],
                    entitlements=list(self._entitlement_cache.values()),
                )

            return ValidationResult(
                valid=False,
                code=result.code,
                message=result.message,
            )

        except Exception as e:
            logger.error(f"License validation failed: {e}")
            return ValidationResult(
                valid=False,
                code="ERROR",
                message=str(e),
            )

    async def is_valid(self) -> bool:
        """
        Quick check if license is valid.

        Returns:
            True if license is valid
        """
        result = await self.validate()
        return result.valid

    # ==================== Entitlement Checking ====================

    async def has_entitlement(
        self,
        feature: str,
        refresh: bool = False,
    ) -> bool:
        """
        Check if license has a specific feature entitlement.

        Args:
            feature: Feature name to check
            refresh: Force refresh from server

        Returns:
            True if feature is entitled and available
        """
        if refresh or feature not in self._entitlement_cache:
            await self.validate(force_refresh=refresh)

        entitlement = self._entitlement_cache.get(feature)
        if not entitlement:
            return False

        return entitlement.is_available

    async def get_entitlement(self, feature: str) -> Optional[Entitlement]:
        """
        Get full entitlement details for a feature.

        Args:
            feature: Feature name

        Returns:
            Entitlement details or None if not found
        """
        if feature not in self._entitlement_cache:
            await self.validate()

        return self._entitlement_cache.get(feature)

    async def get_all_entitlements(self) -> List[Entitlement]:
        """
        Get all entitlements for the license.

        Returns:
            List of all entitlements
        """
        await self.validate()
        return list(self._entitlement_cache.values())

    def require_entitlement(self, feature: str):
        """
        Decorator to require an entitlement for a function.

        Args:
            feature: Required feature name

        Returns:
            Decorator function
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not await self.has_entitlement(feature):
                    raise PermissionError(
                        f"License does not include entitlement: {feature}"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    # ==================== Machine Activation ====================

    async def activate(
        self,
        fingerprint: Optional[str] = None,
        hostname: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActivationResult:
        """
        Activate the license on this machine.

        Args:
            fingerprint: Machine fingerprint (auto-generated if not provided)
            hostname: Machine hostname (auto-detected if not provided)
            metadata: Additional metadata to store with activation

        Returns:
            ActivationResult with activation status
        """
        client = await self._get_client()

        try:
            result = client.activate(
                fingerprint=fingerprint or self._get_fingerprint(),
                hostname=hostname or socket.gethostname(),
                platform=platform.system().lower(),
                metadata=metadata or {},
            )

            if result.success and result.license:
                # Update cached license
                self._cached_license = License(
                    id=result.license.id,
                    key=result.license.key,
                    status=result.license.status,
                    product_code=result.license.product_code,
                    customer_id=result.license.customer_id,
                    tier=result.license.tier,
                    features=result.license.features or [],
                    entitlements=result.license.entitlements or {},
                )
                self._cache_timestamp = datetime.now(timezone.utc)

            return ActivationResult(
                success=result.success,
                machine_id=result.machine_id,
                license=self._cached_license if result.success else None,
                code=result.code,
                message=result.message,
            )

        except Exception as e:
            logger.error(f"Machine activation failed: {e}")
            return ActivationResult(
                success=False,
                code="ERROR",
                message=str(e),
            )

    async def deactivate(
        self,
        fingerprint: Optional[str] = None,
    ) -> bool:
        """
        Deactivate the license from this machine.

        Args:
            fingerprint: Machine fingerprint (uses current machine if not provided)

        Returns:
            True if deactivation successful
        """
        client = await self._get_client()

        try:
            result = client.deactivate(
                fingerprint=fingerprint or self._get_fingerprint()
            )

            if result:
                # Clear cache on deactivation
                self.clear_cache()

            return result

        except Exception as e:
            logger.error(f"Machine deactivation failed: {e}")
            return False

    # ==================== Usage Tracking ====================

    async def record_usage(
        self,
        metric: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageResult:
        """
        Record usage for a metered feature.

        Args:
            metric: Metric name (e.g., "api-calls", "tokens")
            value: Usage value to record
            metadata: Additional metadata

        Returns:
            UsageResult with current usage status
        """
        client = await self._get_client()

        try:
            result = client.record_usage(
                metric=metric,
                value=value,
                metadata=metadata,
            )

            return UsageResult(
                success=result.success,
                metric=metric,
                value_added=result.value_added,
                total_value=result.total_value,
                limit=result.limit,
                remaining=result.remaining,
                code=result.code if hasattr(result, "code") else "",
            )

        except Exception as e:
            logger.error(f"Usage recording failed for {metric}: {e}")
            return UsageResult(
                success=False,
                metric=metric,
                code="ERROR",
            )

    async def heartbeat(
        self,
        version: Optional[str] = None,
    ) -> bool:
        """
        Send heartbeat to keep machine activation alive.

        Args:
            version: Application version

        Returns:
            True if heartbeat successful
        """
        client = await self._get_client()

        try:
            result = client.heartbeat(
                fingerprint=self._get_fingerprint(),
                version=version or "1.0.0",
            )
            return result

        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return False

    # ==================== Utility Methods ====================

    async def get_license_info(self) -> Optional[License]:
        """
        Get current license information.

        Returns:
            License object or None if not validated
        """
        if not self._cached_license:
            await self.validate()
        return self._cached_license

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Vinzy service health.

        Returns:
            Health status dict
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "service": "vinzy",
                            "details": data,
                        }
                    return {
                        "status": "unhealthy",
                        "service": "vinzy",
                        "error": f"HTTP {response.status}",
                    }
        except Exception as e:
            return {
                "status": "unavailable",
                "service": "vinzy",
                "error": str(e),
            }

    def clear_cache(self) -> None:
        """Clear all cached license and entitlement data."""
        self._cached_license = None
        self._cache_timestamp = None
        self._entitlement_cache.clear()
        logger.debug("Cleared Vinzy caches")

    async def close(self) -> None:
        """Close the integration and cleanup resources."""
        if self._client:
            try:
                # LicenseClient may not have async close
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Vinzy client: {e}")
            self._client = None

        self.clear_cache()
        logger.info("VinzyIntegration closed")
