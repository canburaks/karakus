from datetime import datetime
from typing import Dict, Optional

import httpx
from fastapi import Depends, HTTPException

from .config import LemonSqueezySettings, get_lemonsqueezy_settings


class LemonSqueezyService:
    """Service for interacting with LemonSqueezy API."""

    def __init__(self, settings: LemonSqueezySettings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.LEMONSQUEEZY_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.LEMONSQUEEZY_API_KEY}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    async def get_subscription(self, subscription_id: str) -> Dict:
        """Get subscription details."""
        response = await self.client.get(f"/subscriptions/{subscription_id}")
        response.raise_for_status()
        return response.json()["data"]

    async def get_user_tokens(self, user_id: str) -> int:
        """Get available tokens for a user."""
        # In a real implementation, this would query your database
        # For now, we'll return a mock value
        return 100

    async def deduct_tokens(self, user_id: str, tokens: int) -> bool:
        """Deduct tokens from user's balance."""
        # In a real implementation, this would update your database
        # For now, we'll just return True
        return True

    async def verify_webhook_signature(self, signature: str, payload: bytes) -> bool:
        """Verify LemonSqueezy webhook signature."""
        if not self.settings.LEMONSQUEEZY_WEBHOOK_SECRET:
            return False
        # Implement webhook signature verification
        return True

    async def process_webhook_event(self, event_type: str, data: Dict) -> None:
        """Process webhook events from LemonSqueezy."""
        if event_type == "subscription_created":
            # Handle new subscription
            pass
        elif event_type == "subscription_updated":
            # Handle subscription update
            pass
        elif event_type == "subscription_cancelled":
            # Handle subscription cancellation
            pass

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Global instance
_lemonsqueezy_instance: Optional[LemonSqueezyService] = None


async def get_lemonsqueezy_service(
    settings: LemonSqueezySettings = Depends(get_lemonsqueezy_settings),
) -> LemonSqueezyService:
    """Get or create LemonSqueezy service instance."""
    global _lemonsqueezy_instance
    if _lemonsqueezy_instance is None:
        _lemonsqueezy_instance = LemonSqueezyService(settings)
    return _lemonsqueezy_instance
