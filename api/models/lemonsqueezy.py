from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TokenBalance(BaseModel):
    """Model for user token balance."""

    user_id: str
    available_tokens: int
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class TokenTransaction(BaseModel):
    """Model for token transactions."""

    user_id: str
    amount: int  # Positive for purchases, negative for usage
    transaction_type: str  # "purchase" or "usage"
    subscription_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Subscription(BaseModel):
    """Model for LemonSqueezy subscriptions."""

    id: str
    user_id: str
    status: str
    plan_id: str
    tokens_per_period: int
    current_period_end: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookEvent(BaseModel):
    """Model for LemonSqueezy webhook events."""

    event_name: str
    data: Dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
