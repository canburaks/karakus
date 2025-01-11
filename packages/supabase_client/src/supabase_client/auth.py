from datetime import datetime
from typing import Any, List, Optional, TypedDict


class UserIdentity(TypedDict):
    # Add specific identity fields if needed
    pass


class User(TypedDict):
    id: str
    app_metadata: dict[str, Any]
    user_metadata: dict[str, Any]
    aud: str
    confirmation_sent_at: Optional[datetime]
    recovery_sent_at: Optional[datetime]
    email_change_sent_at: Optional[datetime]
    new_email: Optional[str]
    invited_at: Optional[datetime]
    action_link: Optional[str]
    email: str
    phone: str
    created_at: str  # ISO format datetime string
    confirmed_at: Optional[datetime]
    email_confirmed_at: Optional[datetime]
    phone_confirmed_at: Optional[datetime]
    last_sign_in_at: str  # ISO format datetime string
    role: str
    updated_at: str  # ISO format datetime string
    identities: List[UserIdentity]
    factors: Optional[Any]
    is_anonymous: bool


class Session(TypedDict):
    provider_token: Optional[str]
    provider_refresh_token: Optional[str]
    access_token: str
    refresh_token: str
    expires_in: int
    expires_at: int
    token_type: str
    user: User


class AuthResponse(TypedDict):
    user: User
    session: Session
