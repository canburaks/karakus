from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr


class SignUpRequest(BaseModel):
    """
    Request model for user registration.
    
    Attributes:
        email (EmailStr): User's email address
        password (str): User's password
        phone (Optional[str]): Optional phone number
        options (Optional[dict]): Additional signup options
        metadata (Optional[dict]): User metadata
    """
    email: EmailStr
    password: str
    phone: Optional[str] = None
    options: Optional[dict] = None
    metadata: Optional[dict] = None


class SignInRequest(BaseModel):
    """
    Request model for user authentication.
    
    Attributes:
        email (EmailStr): User's email address
        password (str): User's password
        options (Optional[dict]): Additional signin options
        metadata (Optional[dict]): Session metadata
    """
    email: EmailStr
    password: str
    options: Optional[dict] = None
    metadata: Optional[dict] = None


class ResetPasswordRequest(BaseModel):
    email: EmailStr


class UserIdentity(BaseModel):
    # Add specific identity fields if needed
    pass


class TokenRefresh(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str


class SupabaseUser(BaseModel):
    """
    Model representing a Supabase user.
    
    Attributes:
        id (str): Unique user identifier
        app_metadata (Dict[str, Any]): Application-specific metadata
        user_metadata (Dict[str, Any]): User-specific metadata
        email (Optional[str]): User's email address
        phone (Optional[str]): User's phone number
        created_at (datetime): Account creation timestamp
        role (Optional[str]): User's role
        is_anonymous (bool): Whether user is anonymous
    """
    id: str
    app_metadata: Dict[str, Any]
    user_metadata: Dict[str, Any]
    aud: str
    confirmation_sent_at: Optional[datetime] = None
    recovery_sent_at: Optional[datetime] = None
    email_change_sent_at: Optional[datetime] = None
    new_email: Optional[str] = None
    new_phone: Optional[str] = None
    invited_at: Optional[datetime] = None
    action_link: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime
    confirmed_at: Optional[datetime] = None
    email_confirmed_at: Optional[datetime] = None
    phone_confirmed_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    role: Optional[str] = None
    updated_at: Optional[datetime] = None
    identities: Optional[List[UserIdentity]] = None
    is_anonymous: bool = False


class SupabaseSession(BaseModel):
    provider_token: Optional[str] = None
    provider_refresh_token: Optional[str] = None
    access_token: str
    refresh_token: str
    expires_in: int
    expires_at: int = 0
    token_type: str
    user: SupabaseUser


class AuthResponse(BaseModel):
    user: Optional[SupabaseUser] = None
    session: Optional[SupabaseSession] = None
