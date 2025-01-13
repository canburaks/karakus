from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.services.lemonsqueezy import get_lemonsqueezy_service
from api.services.supabase import get_supabase_service

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase_service=Depends(get_supabase_service),
):
    """Get current authenticated user."""
    try:
        user = await supabase_service.client.auth.get_user(credentials.credentials)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


async def verify_token_balance(
    request: Request,
    user=Depends(get_current_user),
    lemonsqueezy_service=Depends(get_lemonsqueezy_service),
):
    """Verify user has sufficient token balance."""
    # Skip token verification for non-AI endpoints
    if not request.url.path.startswith("/ai/"):
        return

    user_id = user.id
    available_tokens = await lemonsqueezy_service.get_user_tokens(user_id)

    # Estimate token usage (this should be refined based on your needs)
    required_tokens = 1

    if available_tokens < required_tokens:
        raise HTTPException(
            status_code=402,
            detail="Insufficient tokens. Please purchase more tokens to continue.",
        )

    # Deduct tokens after successful request
    await lemonsqueezy_service.deduct_tokens(user_id, required_tokens)
