from fastapi import APIRouter, Depends, Header, HTTPException, Request

from api.models.lemonsqueezy import TokenBalance, WebhookEvent
from api.services.lemonsqueezy import get_lemonsqueezy_service

router = APIRouter(
    prefix="/lemonsqueezy",
    tags=["lemonsqueezy"],
    responses={404: {"description": "Not found"}},
)


@router.post("/webhook")
async def handle_webhook(
    request: Request,
    signature: str = Header(..., alias="X-Signature"),
    lemonsqueezy_service=Depends(get_lemonsqueezy_service),
):
    """Handle LemonSqueezy webhook events."""
    payload = await request.body()

    # Verify webhook signature
    if not await lemonsqueezy_service.verify_webhook_signature(signature, payload):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # Parse webhook data
    data = await request.json()
    event = WebhookEvent(
        event_name=data.get("meta", {}).get("event_name"),
        data=data.get("data", {}),
    )

    # Process webhook event
    await lemonsqueezy_service.process_webhook_event(event.event_name, event.data)

    return {"status": "success"}


@router.get("/tokens/{user_id}")
async def get_token_balance(
    user_id: str,
    lemonsqueezy_service=Depends(get_lemonsqueezy_service),
) -> TokenBalance:
    """Get token balance for a user."""
    available_tokens = await lemonsqueezy_service.get_user_tokens(user_id)
    return TokenBalance(user_id=user_id, available_tokens=available_tokens)
