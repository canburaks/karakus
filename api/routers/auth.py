from fastapi import APIRouter, Depends, HTTPException

from api.services.supabase import get_auth_service, get_supabase_client

from ..core.logger import log
from ..models.auth import (
    AuthResponse,
    ResetPasswordRequest,
    SignInRequest,
    SignUpRequest,
    SupabaseUser,
)

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    dependencies=[],
    responses={404: {"Users": "Not found"}},
)


@router.post("/signup", response_model=AuthResponse)
async def sign_up(
    request: SignUpRequest, client=Depends(get_auth_service)
) -> AuthResponse:
    try:
        return client.client.auth.sign_up(
            {
                "email": request.email,
                "password": request.password,
                "phone": request.phone,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/signin", response_model=AuthResponse)
async def sign_in(
    request: SignInRequest, client=Depends(dependency=get_auth_service)
) -> AuthResponse:
    try:
        response = client.client.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )
        log.info(f"Sign in response: {response}")
        return response
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/signout")
async def sign_out(client=Depends(dependency=get_auth_service)):
    try:
        await client.client.auth.sign_out()
        return {"message": "Successfully signed out"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest, client=Depends(dependency=get_auth_service)
):
    try:
        await client.client.auth.reset_password_for_email(request.email)
        return {"message": "Password reset email sent"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/user", response_model=SupabaseUser)
async def get_user(client=Depends(dependency=get_auth_service)) -> SupabaseUser:
    try:
        user = await client.client.auth.get_user()
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Not authenticated")
